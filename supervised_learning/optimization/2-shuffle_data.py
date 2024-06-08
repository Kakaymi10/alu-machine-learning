#!/usr/bin/env python3
'''
optimization
'''


import numpy as np


def shuffle_data(X,Y):
    '''
    returns shuffled matrices
    '''
    a = np.random.permutation(X)
    b = np.random.permutation(Y)
    return (a,b)
