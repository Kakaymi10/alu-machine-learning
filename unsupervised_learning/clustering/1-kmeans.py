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
    # Validate input types and values
    if not isinstance(X, np.ndarray) or X.ndim != 2 or not isinstance(k, int) or k <= 0:
        return None, None
    if not isinstance(iterations, int) or iterations <= 0:
        return None, None

    n, d = X.shape

    # Initialize centroids randomly within data range
    low, high = np.amin(X, axis=0), np.amax(X, axis=0)
    C = np.random.uniform(low, high, size=(k, d))

    for _ in range(iterations):
        # Assign each data point to the nearest centroid
        distances = np.linalg.norm(X[:, None] - C, axis=-1)
        clss = np.argmin(distances, axis=-1)

        # Update centroids based on cluster assignments
        new_C = np.array([X[clss == c].mean(axis=0) if np.any(clss == c) 
                          else np.random.uniform(low, high, size=(d,)) 
                          for c in range(k)])

        # Check for convergence
        if np.allclose(C, new_C):
            break
        C = new_C

    # Final cluster assignments
    clss = np.argmin(np.linalg.norm(X[:, None] - C, axis=-1), axis=-1)
    return C, clss

