#!/usr/bin/env python3
"""Calculating Total Intra-Cluster Variance"""


import numpy as np


def variance(X, C):
    """
    Calculates the total intra-cluster variance for a data set.

    Parameters:
        X (numpy.ndarray): Data set of shape (n, d).
        C (numpy.ndarray): Centroid means of shape (k, d).

    Returns:
        var (float): The total variance, or None on failure.
    """
    if not (isinstance(X, np.ndarray) and isinstance(C, np.ndarray)):
        return None
    if X.ndim != 2 or C.ndim != 2 or X.shape[1] != C.shape[1]:
        return None

    # Calculate the distance from each point in X to all centroids in C
    distances = np.linalg.norm(X[:, None] - C, axis=-1)

    # Assign each data point to the nearest centroid
    min_distances = np.min(distances, axis=1)

    # Compute the total intra-cluster variance
    var = np.sum(min_distances ** 2)

    return var