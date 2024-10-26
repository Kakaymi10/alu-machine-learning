#!/usr/bin/env python3
'''
Viterbi
'''


import numpy as np


def viterbi(Observation, Emission, Transition, Initial):
    '''
    Determines the most likely sequence of
    hidden states for a hidden Markov model
    '''
    # Number of hidden states
    N = Transition.shape[0]

    # Number of observations
    T = len(Observation)

    # Viterbi matrix to store the probabilities
    V = np.zeros((N, T))

    # Backpointer matrix to store the most likely previous state
    backpointer = np.zeros((N, T), dtype=int)

    # Initialization step
    for i in range(N):
        V[i, 0] = Initial[i, 0] * Emission[i, Observation[0]]

    # Recursion step
    for t in range(1, T):
        for i in range(N):
            max_prob = -1
            max_state = 0

            for j in range(N):
                prob = V[j, t - 1] * Transition[j, i]

                if prob > max_prob:
                    max_prob = prob
                    max_state = j

            V[i, t] = max_prob * Emission[i, Observation[t]]
            backpointer[i, t] = max_state

    # Termination step
    best_path_prob = -1
    best_last_state = 0
    for i in range(N):
        if V[i, T - 1] > best_path_prob:
            best_path_prob = V[i, T - 1]
            best_last_state = i

    # Backtrack to find the best path
    path = [0] * T
    path[T - 1] = best_last_state

    for t in range(T - 2, -1, -1):
        path[t] = backpointer[path[t + 1], t + 1]

    return path, best_path_prob
