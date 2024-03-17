#!/usr/bin/env python3
"""
This module contains a function to calculate
the correlation matrix.
"""

import numpy as np


def correlation(C):
    """
    Calculate the correlation matrix.

    Args:
        C (numpy.ndarray): The input covariance matrix of
        shape (d, d), where d is the number of dimensions.

    Returns:
        numpy.ndarray: The correlation matrix of shape (d, d).
    """
    if not isinstance(C, np.ndarray):
        raise TypeError("C must be a numpy.ndarray")
    if C.ndim != 2 or C.shape[0] != C.shape[1]:
        raise ValueError("C must be a 2D square matrix")

    d = C.shape[0]
    correlation_matrix = np.zeros((d, d))
    for i in range(d):
        for j in range(d):
            correlation_matrix[i, j] = C[i, j] / np.sqrt(C[i, i] * C[j, j])

    return correlation_matrix
