#!/usr/bin/env python3
"""Performing K-means on a dataset"""


import numpy as np


def kmeans(X, k, iterations=1000):
    """
    Performs K-means clustering on a dataset.

    Parameters:
        X (numpy.ndarray): Dataset of shape (n, d).
        k (int): Number of clusters.
        iterations (int): Maximum number of iterations (default: 1000).

    Returns:
        tuple: (C, clss) or (None, None) on failure.
            - C (numpy.ndarray): Centroid means of shape (k, d).
            - clss (numpy.ndarray): Cluster index for each data
            point of shape (n,).
    """
    if not (isinstance(X, np.ndarray) and X.ndim == 2
            and isinstance(k, int) and k > 0):
        return None, None
    if not (isinstance(iterations, int) and iterations > 0):
        return None, None

    n, d = X.shape
    low, high = np.amin(X, axis=0), np.amax(X, axis=0)
    C = np.random.uniform(low, high, size=(k, d))

    for _ in range(iterations):
        clss = np.argmin(np.linalg.norm(X[:, None] - C, axis=-1), axis=-1)
        new_C = np.array([X[clss == i].mean(axis=0) if np.any(clss == i)
                          else np.random.uniform(low, high, size=d)
                          for i in range(k)])

        if np.allclose(C, new_C):
            break
        C = new_C

    clss = np.argmin(np.linalg.norm(X[:, None] - C, axis=-1), axis=-1)
    return C, clss
