#!/usr/bin/env python3
"""Testing for the optimum number of clusters"""

import numpy as np
kmeans = __import__('1-kmeans').kmeans
variance = __import__('2-variance').variance


def optimum_k(X, kmin=1, kmax=None, iterations=1000):
    """Function that tests for the optimum number of clusters
    X is a numpy.ndarray of shape (n, d) containing the data set
    kmin is a positive integer containing the minimum
        number of clusters to check for (inclusive)
    kmax is a positive integer containing the maximum
        number of clusters to check for (inclusive)
    iterations is a positive integer containing the maximum
        number of iterations for K-means
    This function should analyze at least 2 different cluster sizes
    """
    try:
        if not isinstance(X, np.ndarray) or len(X.shape) != 2:
            return None, None
        if iterations <= 0 or not isinstance(iterations, int):
            return None, None
        if kmax is not None and not isinstance(kmax, int) or kmax <= 0:
            return None, None
        if not kmax:
            kmax = X.shape[0]
        if not isinstance(kmin, int) or kmin >= kmax or kmin <= 0:
            return None, None

        result, df_var = [], []
        n, d = X.shape

        for k in range(kmin, kmax + 1):
            C, clss = kmeans(X, k, iterations)
            result.append((C, clss))
            df_var.append(variance(X, C))

        df_var = [df_var[0] - x for x in df_var]

        return result, df_var

    except Exception:
        return None, None