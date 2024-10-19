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
    calculates intra-cluster variance for a dataset

    X: numpy.ndarray (n, d) containing the dataset
        - n no. of data points
        - d no. of dimensions for each data point
    kmin: positive integer - the minimum no. of clusters
    kmax: positive integer - the maximum no. of clusters
    iterations: +ve(int) - max no. of iterations performed

    return:
        - results: list containing the results of the
        K-means for each cluster size
        - d_vars: list containing the difference in variance
        from the smallest cluster size for each cluster size
    """
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
    var = float('inf')
    
    for k in range(kmin, kmax + 1):
        C, clss = kmeans(X, k, iterations)
        
        # Check if kmeans returned valid results
        if C is None or clss is None:
            print(f"Warning: kmeans failed for k={k}")
            continue
        
        results.append((C, clss))
        new_var = variance(X, C)
        
        # Check if variance calculation returned valid result
        if new_var is None:
            print(f"Warning: variance returned None for k={k}")
            continue
        
        if k == kmin:
            var = new_var
        
        d_vars.append(var - new_var)

    return results, d_vars
