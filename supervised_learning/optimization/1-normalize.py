#!/usr/bin/env python3
'''
optimization
'''


import numpy as np


def normalize(X, m, s):
    '''
    return normalized X
    '''
    for i in range(len(X)):
        for j in range(len(X[0])):
            X[i][j] = (X[i][j] - m[j])/s[j]
    return X
