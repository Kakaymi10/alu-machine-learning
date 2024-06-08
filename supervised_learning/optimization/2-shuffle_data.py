#!/usr/bin/env python3
'''
optimization
'''


import numpy as np


def shuffle_data(X, Y):
    '''
    returns shuffled matrices
    '''
    s = np.random.permutation(X.shape[0])
    return X[s], Y[s]
