#!/usr/bin/env python3
'''
optimiozation
'''

import numpy as np


def normalization_constants(X):
    '''
    return the mean and standard deviation
    '''
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    return (mean, std)
