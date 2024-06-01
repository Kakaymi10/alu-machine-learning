#!/usr/bin/env python3
""" Forward Propagation with Dropout """


import numpy as np


def dropout_forward_prop(X, weights, L, keep_prob):
    """
    Perform forward propagation with dropout regularization.
    """
    cache = {'A0': X}
    for i in range(1, L + 1):
        W = weights[f'W{i}']
        b = weights[f'b{i}']
        A_prev = cache[f'A{i - 1}']
        Z = np.matmul(W, A_prev) + b
        
        if i < L:
            A = np.tanh(Z)
            D = np.random.rand(*A.shape) < keep_prob
            A = np.multiply(A, D) / keep_prob
            cache[f'D{i}'] = D
        else:
            A = np.exp(Z) / np.sum(np.exp(Z), axis=0, keepdims=True)
        
        cache[f'A{i}'] = A
    
    return cache
