#!/usr/bin/env python3
"""Performing K-means on a dataset."""

import numpy as np

def kmeans(X, k, iterations=1000):
    """
    Performs K-means clustering on a dataset.

    Parameters:
        X (numpy.ndarray): The dataset of shape (n, d).
        k (int): The number of clusters.
        iterations (int): Maximum number of iterations (default: 1000).

    Returns:
        C (numpy.ndarray): The centroid means for each cluster of shape (k, d).
        clss (numpy.ndarray): Cluster assignments of shape (n,).
        Returns (None, None) on failure.
    """
    # Input validation
    if (
        not isinstance(X, np.ndarray) or X.ndim != 2 or
        not isinstance(k, int) or k <= 0 or
        not isinstance(iterations, int) or iterations <= 0
    ):
        return None, None

    n, d = X.shape

    # Initialize centroids using uniform distribution
    low = np.amin(X, axis=0)
    high = np.amax(X, axis=0)
    C = np.random.uniform(low, high, size=(k, d))

    for _ in range(iterations):
        # Compute distances and assign each point to the nearest cluster
        distances = np.linalg.norm(X[:, None] - C, axis=-1)
        clss = np.argmin(distances, axis=-1)

        # Update centroids
        new_C = np.copy(C)
        for c in range(k):
            points_in_cluster = X[clss == c]
            if points_in_cluster.size == 0:  # Reinitialize empty cluster
                new_C[c] = np.random.uniform(low, high, size=(d,))
            else:
                new_C[c] = np.mean(points_in_cluster, axis=0)

        # Check for convergence
        if np.allclose(C, new_C):
            break
        C = new_C

    return C, clss
