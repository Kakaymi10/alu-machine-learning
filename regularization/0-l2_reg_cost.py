#!/usr/bin/env python3
'''
regularize me
'''

import numpy as np


def l2_reg_cost(cost, lambtha, weights, L, m):
    '''
    calculates the cost of a neural network with L2 regularization
    '''
    regularizer = 0
    for i in range(1, L+1):
        w = weights["W"+str(i)]
        regularizer += np.sum(np.square(w))
    new_cost = cost + regularizer
    return new_cost
