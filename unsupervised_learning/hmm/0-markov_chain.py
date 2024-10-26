#!/usr/bin/env python3
'''
Markov Chain
'''


import numpy as np


def markov_chain(P, s, t=1):
    '''
    Function that determines the probability of a markov chain being in a
    particular state after a specified number of iterations
    '''
    # Step 1: Validate the inputs
    if not isinstance(P, np.ndarray) or not isinstance(s, np.ndarray):
        return None
    if P.shape[0] != P.shape[1] or s.shape != (1, P.shape[0]):
        return None

    # Step 2: Calculate P^t (P to the power of t)
    try:
        P_t = np.linalg.matrix_power(P, t)
    except np.linalg.LinAlgError:
        return None  # If matrix power calculation fails

    # Step 3: Multiply the initial state vector s by P^t
    result = np.matmul(s, P_t)

    return result
