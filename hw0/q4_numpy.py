"""Code for HW0 Problem 4: numpy & gradient descent."""
import argparse
import sys

import numpy as np

v_1 = np.array([1, 4])
v_2 = np.array([1, 0])

def f(x):
    ### BEGIN_SOLUTION 4a
    ## f(x) = −vT1*x + ∥x − v2∥^2
    first = np.dot(-1*v_1.T,x)
    second = np.linalg.norm(np.subtract(x,v_2))**2
    return np.add(first,second)
    ### END_SOLUTION 4a

    ##Find the derivative of f(x) 
def grad_f(x):
    ### BEGIN_SOLUTION 4c
    first = -1*v_1.T
    second = 2*np.subtract(x,v_2)
    return np.add(first,second)
    ### END_SOLUTION 4c
    

def find_optimum(x):
    ### BEGIN_SOLUTION 4d
    learning_rate=1e-1
    num_iters=100
    for i in range(num_iters):
        x = x - (learning_rate * grad_f(x))
        print(f'x" {x}, f_x: {f(x)}')
    return x
    ### END_SOLUTION 4d

def main():
    x = find_optimum(np.zeros(2))
    print('Optimal x: ', x)
    print('Optimal f(x): ', f(x))

if __name__ == '__main__':
    main()

