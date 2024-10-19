#!/usr/bin/env python3

"""
This module contains a function that
tests for the optimum number of clusters by variance
"""

import numpy as np
kmeans = __import__('1-kmeans').kmeans
variance = __import__('2-variance').variance


def optimum_k(X, kmin=1, kmax=None, iterations=1000):
    """
    Calculates intra-cluster variance for a dataset.

    X: numpy.ndarray (n, d) containing the dataset
        - n: number of data points
        - d: number of dimensions for each data point
    kmin: positive integer - the minimum number of clusters
    kmax: positive integer - the maximum number of clusters
    iterations: positive integer - max number of iterations performed

    return:
        - results: list containing the results of the
        K-means for each cluster size
        - d_vars: list containing the difference in variance
        from the smallest cluster size for each cluster size
    """
    # Check input validity
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None, None
    if not isinstance(kmin, int) or kmin <= 0:
        return None, None
    if kmax is not None and (not isinstance(kmax, int) or kmax <= 0):
        return None, None
    if kmin >= (kmax if kmax is not None else float('inf')):
        return None, None
    if not isinstance(iterations, int) or iterations <= 0:
        return None, None
    if kmax is None:
        kmax = X.shape[0]

    results = []
    d_vars = []
    variances = np.zeros(kmax - kmin + 1)  # Store variances for each k

    # Single loop to compute the K-means and variance
    for k in range(kmin, kmax + 1):
        C, clss = kmeans(X, k, iterations)

        # Ensure kmeans returned valid results
        if C is None or clss is None:
            print(f"Warning: kmeans failed for k={k}")
            continue
        
        results.append((C, clss))
        variances[k - kmin] = variance(X, C)  # Store the variance for this k

    # Compute differences in variance based on the smallest variance found
    min_variance = np.min(variances[variances > 0])  # Find minimum positive variance
    d_vars = min_variance - variances

    return results, d_vars.tolist()  # Convert d_vars to list for consistent return type
