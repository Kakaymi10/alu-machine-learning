#!/usr/bin/env python3
'''
backward
'''


import numpy as np


def backward(Observation, Emission, Transition, Initial):
    '''
    Performs the backward algorithm for a hidden Markov model
    '''
    # Number of hidden states
    N = Transition.shape[0]

    # Number of observations
    T = len(Observation)

    # Backward probabilities matrix
    B = np.zeros((N, T))

    # Initialization step: at time T-1, the backward probabilities are all 1
    B[:, T - 1] = 1

    # Recursion step: fill in the backward probabilities
    for t in range(T - 2, -1, -1):
        for i in range(N):
            total = 0
            for j in range(N):
                a = Transition[i, j]
                b = Emission[j, Observation[t + 1]]
                c = B[j, t + 1]
                total += a * b * c
            B[i, t] = total

    # Likelihood of the observations given the model
    P = 0
    for i in range(N):
        P += Initial[i, 0] * Emission[i, Observation[0]] * B[i, 0]

    return P, B
