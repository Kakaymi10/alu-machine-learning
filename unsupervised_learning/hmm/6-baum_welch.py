#!/usr/bin/env python3
""" baum_welch algorithm"""

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


def backward(Observation, Emission, Transition, Initial):
    """
    Performs the backward algorithm for a hidden Markov model
    """
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


def baum_welch(Observations, Transition, Emission, Initial, iterations=1000):
    """
    Function that performs the Baum-Welch algorithm for a hidden Markov model
    """
    N = Transition.shape[0]
    M = Emission.shape[1]
    T = Observations.shape[0]

    for _ in range(iterations):
        _, F = forward(Observations, Emission, Transition, Initial)
        _, B = backward(Observations, Emission, Transition, Initial)

        xi = np.zeros((N, N, T - 1))
        for t in range(T - 1):
            denominator = np.dot(np.dot(F[:, t].T, Transition) *
                                 Emission[:, Observations[t + 1]] *
                                 B[:, t + 1].T, 1)
            for i in range(N):
                numerator = F[i, t] * Transition[i] * \
                    Emission[:, Observations[t + 1]] * B[:, t + 1].T
                xi[i, :, t] = numerator / denominator

        gamma = np.sum(xi, axis=1)
        Transition = np.sum(xi, 2) / np.sum(gamma, axis=1).reshape((-1, 1))

        # Add the last gamma
        gamma = np.hstack((gamma,
                           np.sum(xi[:, :, T - 2], axis=0).reshape((-1, 1))))

        denominator = np.sum(gamma, axis=1)
        for i in range(M):
            Emission[:, i] = np.sum(gamma[:, Observations == i], axis=1)

        Emission = np.divide(Emission, denominator.reshape((-1, 1)))

    return Transition, Emission
