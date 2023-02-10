"""Code for HW1 Problem 4: for Author Attribution with Naive Bayes."""
import argparse
from collections import Counter, defaultdict
import sys
from tqdm import tqdm
import numpy as np

OPTS = None

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--evaluation-set', '-e', choices=['dev', 'test', 'newbooks'])
    parser.add_argument('--analyze-counts', '-a', action='store_true')
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    return parser.parse_args()

def read_data(filename):
    dataset = []
    with open(filename, 'r', encoding='utf-8') as f:
        for line in f:
            label, book, passage = line.strip().split('\t')
            dataset.append((passage.split(' '), label))
    return dataset

def get_vocabulary(dataset):
    return list(set(word for (words, label) in dataset for word in words))

def get_label_counts(train_data):
    """Count the number of examples with each label in the dataset.

    We will use a Counter object from the python collections library.
    A Counter is essentially a dictionary with a "default value" of 0
    for any key that hasn't been inserted into the dictionary.

    Args:
        train_data: A list of (words, label) pairs, where words is a list of str
    Returns:
        A Counter object mapping each label to a count.
    """
    label_counts = Counter()
    ### BEGIN_SOLUTION 4a
    for words, label in train_data:
        label_counts[label] += len(words)
    ### END_SOLUTION 4a
    return label_counts

def get_word_counts(train_data):
    """Count occurrences of every word with every label in the dataset.

    We will create a separate Counter object for each label.
    To do this easily, we create a defaultdict(Counter),
    which is a dictionary that will create a new Counter object whenever
    we query it with a key that isn't in the dictionary.

    Args:
        train_data: A list of (words, label) pairs, where words is a list of str
    Returns:
        A Counter object where keys are tuples of (label, word), mapped to
        the number of times that word appears in an example with that label
    """
    word_counts = defaultdict(Counter)
    ### BEGIN_SOLUTION 4a
    for words, label in train_data:
        for word in words:
            word_counts[label][word] += 1
    ### END_SOLUTION 4a
    return word_counts

def predict(words, label_counts, word_counts, vocabulary):
    """Return the most likely label given the label_counts and word_counts.

    Args:
        words: List of words for the current input.
        label_counts: Counts for each label in the training data
        word_counts: Counts for each (label, word) pair in the training data
        vocabulary: List of all words in the vocabulary
    Returns:
        The most likely label for the given input words.
    """
    labels = list(label_counts.keys())  # A list of all the labels
    n = len(labels)
    d = len(words)
    # 
    ### BEGIN_SOLUTION 4a
    # Step 1: Parameter Estimation training
    #lamda laplace smoothing param
    lamda = 1
    #P(author)= #author text / # total text, turned to log prior probability early
    log_prior = np.zeros(n)
    total_labels = sum(label_counts.values())
    for idx, label in enumerate(labels):
        log_prior[idx] = np.log(label_counts[label]/total_labels)
    #p(word | author) = # times word appeared with label / total words per label, turned to log likelihood early
    log_ll = np.zeros((n,d))
    #total_words = sum(word_counts.values())
    for idx_w, word in enumerate(set(words)):
        for idx_l, label in enumerate(labels):
            ## Likelihood with laplace smoothing, denominator is lamda * # of vocab/unique words in set
            log_ll[idx_l][idx_w] = np.log((word_counts[label][word] + lamda)/(label_counts[label] + (len(vocabulary) * lamda)))
    
    # Step 2: Inference
    # Sum the log likelihoods of word given author
    log_ll_sum = np.sum(log_ll, axis = 1)

    # Add the probabilities together for each label and choose largest
    log_post = log_prior+log_ll_sum
    label_prediction = np.argmax(log_post)
    return labels[label_prediction]
    ### END_SOLUTION 4a

def evaluate(label_counts, word_counts, vocabulary, dataset, name, print_confusion_matrix=False):
    num_correct = 0
    confusion_counts = Counter()
    for words, label in tqdm(dataset, desc=f'Evaluating on {name}'):
        pred_label = predict(words, label_counts, word_counts, vocabulary)
        confusion_counts[(label, pred_label)] += 1
        if pred_label == label:
            num_correct += 1
    accuracy = 100 * num_correct / len(dataset)
    print(f'{name} accuracy: {num_correct}/{len(dataset)} = {accuracy:.3f}%')
    if print_confusion_matrix:
        print(''.join(['actual\\predicted'] + [label.rjust(12) for label in label_counts]))
        for true_label in label_counts:
            print(''.join([true_label.rjust(16)] + [
                    str(confusion_counts[true_label, pred_label]).rjust(12)
                    for pred_label in label_counts]))

def analyze_counts(label_counts, word_counts, vocabulary):
    """Analyze the word counts to identify the most predictive features.

    For each label, print out the top ten words that are most indictaive of the label.
    There are multiple valid ways to define what "most indicative" means.
    Our definition is that if you treat the single word as the input x,
    and assume a uniform prior over the labels,
    find the words with largest p(y=label | x).
    """
    labels = list(label_counts.keys())  # A list of all the labels
    ### BEGIN_SOLUTION 4b
    n = len(labels)
    lamda = 1
    #P(author)= #author text / # total text
    prior_p = np.zeros(n)
    total_labels = sum(label_counts.values())
    for idx, label in enumerate(labels):
        prior_p[idx] = label_counts[label]/total_labels
    #p(word | author) = # times word appeared with label / total words per label
    numerator = {}
    denominator = {}
    word_assoc = {}
    #total_words = sum(word_counts.values())
    for word in vocabulary:
        #Marginalization P(word)
        word_sum = 0
        for idx_l, label in enumerate(labels):
            ## Likelihood with laplace smoothing, denominator is lamda * # of vocab/unique words in set
            ll = (word_counts[label][word] + lamda)/(label_counts[label] + (len(vocabulary) * lamda))
            numerator[(word,label)] = prior_p[idx_l] * ll
            word_sum += prior_p[idx_l] * ll
        denominator[word] = word_sum
    for word in vocabulary:
        for label in labels:
            word_assoc[(word,label)] = numerator[(word,label)]/denominator[word]
            #print(f'{label}|{word}: {word_assoc[word,label]}')
        
    ranking = sorted(word_assoc.items(), key=lambda x: x[1])
    austen, christie, melville, shakespeare = [], [], [], []
    author_to_list = {
        'austen': austen,
        'christie': christie,
        'melville': melville,
        'shakespeare': shakespeare
    }

    while (len(austen) < 10 or len(christie) < 10 or len(melville) < 10 or len(shakespeare) < 10):
        ((word, label), _) = ranking.pop()
        label_arr = author_to_list[label]
        if (len(label_arr) >= 10):
            continue
        label_arr.append((word,word_assoc[(word,label)]))

    for author, words in author_to_list.items():
        print(f'{author}: {words}')
        print('\n')
    ### END_SOLUTION 4b

def main():
    train_data = read_data('train.tsv')
    dev_data = read_data('dev.tsv')
    test_data = read_data('test.tsv')
    newbooks_data = read_data('newbooks.tsv')

    vocabulary = get_vocabulary(train_data)  # The set of words present in the training data
    label_counts = get_label_counts(train_data)
    word_counts = get_word_counts(train_data)
    if OPTS.analyze_counts:
        analyze_counts(label_counts, word_counts, vocabulary)
    evaluate(label_counts, word_counts, vocabulary, train_data, 'train')
    if OPTS.evaluation_set == 'dev':
        evaluate(label_counts, word_counts, vocabulary, dev_data, 'dev', print_confusion_matrix=True)
    elif OPTS.evaluation_set == 'test':
        evaluate(label_counts, word_counts, vocabulary, test_data, 'test', print_confusion_matrix=True)
    elif OPTS.evaluation_set == 'newbooks':
        evaluate(label_counts, word_counts, vocabulary, newbooks_data, 'newbooks', print_confusion_matrix=True)

if __name__ == '__main__':
    OPTS = parse_args()
    main()

