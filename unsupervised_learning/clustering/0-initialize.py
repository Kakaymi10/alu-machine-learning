#!/usr/bin/env python3
'''Initialize K-means'''


import numpy as np


def initialize(X, k):
    """Initialize cluster centroids for K-means using uniform distribution.

    Args:
        X (numpy.ndarray): Dataset of shape (n, d) for K-means clustering.
        k (int): Number of clusters.

    Returns:
        numpy.ndarray or None: Initialized centroids of shape (k, d),
        or None on failure.
    """
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None
    if not isinstance(k, int) or k <= 0:
        return None

    n, d = X.shape
    min_vals = X.min(axis=0)
    max_vals = X.max(axis=0)

    # Generate k centroids using a uniform distribution
    centroids = np.random.uniform(min_vals, max_vals, size=(k, d))
    return centroids
