#!/usr/bin/env python3
'''K-means clustering'''


import numpy as np


def initialize(X, k):
    """Initialize cluster centroids for K-means using uniform distribution."""
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None
    if not isinstance(k, int) or k <= 0:
        return None

    n, d = X.shape
    min_vals = X.min(axis=0)
    max_vals = X.max(axis=0)

    centroids = np.random.uniform(min_vals, max_vals, size=(k, d))
    return centroids

def kmeans(X, k, iterations=1000):
    """Performs K-means clustering on a dataset."""
    if (
        not isinstance(X, np.ndarray) or len(X.shape) != 2 or
        not isinstance(k, int) or k <= 0 or
        not isinstance(iterations, int) or iterations <= 0
    ):
        return None, None

    n, d = X.shape
    C = initialize(X, k)  # Initialize centroids
    if C is None:
        return None, None

    clss = np.zeros(n, dtype=int)  # Cluster assignments

    for _ in range(iterations):
        # Compute distances and assign clusters
        distances = np.linalg.norm(X[:, np.newaxis] - C, axis=2)
        new_clss = np.argmin(distances, axis=1)

        # If no change in cluster assignments, exit early
        if np.array_equal(clss, new_clss):
            break
        clss = new_clss

        # Update centroids
        for i in range(k):
            points_in_cluster = X[clss == i]
            if points_in_cluster.size == 0:  # Reinitialize if no points in cluster
                C[i] = np.random.uniform(X.min(axis=0), X.max(axis=0), size=(d,))
            else:
                C[i] = points_in_cluster.mean(axis=0)

    return C, clss
