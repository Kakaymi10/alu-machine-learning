#!/usr/bin/env python3
'''
Regular Chains
'''


import numpy as np


def regular(P):
    '''
    Function that determines the steady state probabilities of a regular
    markov chain
    '''

    # Step 1: Validate the inputs
    if not isinstance(P, np.ndarray):
        return None
    if P.shape[0] != P.shape[1]:
        return None
    if np.any(P <= 0):
        return None
    if not np.all(np.isclose(P.sum(axis=1), 1)):
        return None

    # Step 2: Calculate the steady state probabilities
    n = P.shape[0]
    Im = np.eye(n)
    P_t = np.transpose(P) - Im
    P_t[-1] = np.ones(n)
    b = np.zeros(n)
    b[-1] = 1
    x = np.linalg.solve(P_t, b)
    x = x / np.sum(x)

    return x.reshape(1, n)
