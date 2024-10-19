#!/usr/bin/env python3
"""Determines the optimal number of clusters
using variance."""


import numpy as np
kmeans = __import__('1-kmeans').kmeans
variance = __import__('2-variance').variance


def optimum_k(X, kmin=1, kmax=None, iterations=1000):
    """Determines the optimal number of clusters using variance."""
    if kmax is None or kmax < kmin:
        return None, None

    results = []
    d_vars = []

    for k in range(kmin, kmax + 1):
        C, clss = kmeans(X, k, iterations)
        if C is None:
            return None, None

        results.append((C, clss))
        var = variance(X, C)
        d_vars.append(var)

    # Compute the difference in variance relative to kmin
    d_vars = [d_vars[0] - v for v in d_vars]

    return results, d_vars
