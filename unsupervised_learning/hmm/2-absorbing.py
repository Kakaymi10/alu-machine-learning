#!/usr/bin/env python3
'''
Absorbing Chains
'''


import numpy as np


def absorbing(P):
    '''
    Function that determines if a markov chain is absorbing
    '''
    # Check if P is a square matrix
    if not isinstance(P, np.ndarray) or P.shape[0] != P.shape[1]:
        return False

    n = P.shape[0]

    # Step 1: Identify absorbing states
    absorbing_states = np.where(np.diag(P) == 1)[0]

    # If there are no absorbing states, return False
    if len(absorbing_states) == 0:
        return False

    # Step 2: Check reachability for non-absorbing states
    # Create a reachability matrix using powers of P
    reachability_matrix = np.copy(P)
    for _ in range(n - 1):
        reachability_matrix = reachability_matrix @ P  # matrix multiplication

    # For each non-absorbing state, check if it can reach an absorbing state
    for i in range(n):
        if i not in absorbing_states:
            # Check if there's a non-zero probability to an absorbing state
            if not any(reachability_matrix[i, j] > 0
                       for j in absorbing_states):
                return False

    return True
