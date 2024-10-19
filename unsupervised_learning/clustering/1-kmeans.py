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
            - clss (numpy.ndarray): Cluster index for each data point of shape (n,).
    """
    # Validate input
    if not isinstance(X, np.ndarray) or X.ndim != 2:
        return None, None
    if not isinstance(k, int) or k <= 0 or not isinstance(iterations, int) or iterations <= 0:
        return None, None

    n, d = X.shape

    # Initialize centroids randomly within data range
    low, high = np.amin(X, axis=0), np.amax(X, axis=0)
    C = np.random.uniform(low, high, size=(k, d))

    for _ in range(iterations):
        # Assign data points to the nearest centroid
        distances = np.linalg.norm(X[:, None] - C, axis=-1)
        clss = np.argmin(distances, axis=-1)

        # Update centroids (one loop for k clusters)
        new_C = np.empty_like(C)
        for c in range(k):
            points = X[clss == c]
            if len(points) == 0:
                new_C[c] = np.random.uniform(low, high, size=(d,))
            else:
                new_C[c] = points.mean(axis=0)

        # Check for convergence
        if np.allclose(C, new_C):
            break
        C = new_C

    # Final cluster assignments
    clss = np.argmin(np.linalg.norm(X[:, None] - C, axis=-1), axis=-1)
    return C, clss