#!/usr/bin/env python3
'''
Regular Chains
'''


import numpy as np


def forward(Observation, Emission, Transition, Initial):
    """
    Performs the forward algorithm for a hidden Markov model.
    """
    T = Observation.shape[0]
    N, M = Emission.shape

    F = np.zeros((N, T))
    F[:, 0] = Initial.T * Emission[:, Observation[0]]

    for t in range(1, T):
        for n in range(N):
            F[n, t] = np.sum(F[:, t-1] *
                             Transition[:, n] *
                             Emission[n, Observation[t]])

    P = np.sum(F[:, -1])
    return P, F
