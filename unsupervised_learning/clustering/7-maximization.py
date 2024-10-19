#!/usr/bin/env python3
"""Calculating the maximization step in the EM algorithm for a GMM"""

import numpy as np


def maximization(X, g):
    """Function that calculates the maximization
        step in the EM algorithm for a GMM
    X is a numpy.ndarray of shape (n, d) containing the data set
    g is a numpy.ndarray of shape (k, n) containing the
        posterior probabilities for each data point in each cluster

    Returns: pi, m, S, or None, None, None on failure
        pi is a numpy.ndarray of shape (k,)
            containing the updated priors for each cluster
        m is a numpy.ndarray of shape (k, d)
            containing the updated centroid means for each cluster
        S is a numpy.ndarray of shape (k, d, d)
            containing the updated covariance matrices for each cluster
    """
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None, None, None
    if not isinstance(g, np.ndarray) or len(g.shape) != 2:
        return None, None, None
    n, d = X.shape
    k = g.shape[0]
    if n != g.shape[1]:
        return None, None, None
    if not np.isclose(np.sum(g, axis=0), 1).all():
        return None, None, None
    pi = np.sum(g, axis=1) / n
    m = np.dot(g, X) / np.sum(g, axis=1)[:, np.newaxis]
    S = np.zeros((k, d, d))
    for i in range(k):
        y = X - m[i]
        S[i] = np.dot(g[i] * y.T, y) / np.sum(g[i])
    return pi, m, S
