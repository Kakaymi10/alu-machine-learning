#!/usr/bin/env python3
"""
Deep Neural Network Class
"""


import numpy as np


def one_hot_encode(Y, classes):
    """
    Converts a numeric label vector into a one-hot matrix.
    """
    # Check if Y is a numpy ndarray and has the correct shape
    if not isinstance(Y, np.ndarray) or Y.ndim != 1:
        return None

    try:
        # Number of examples
        m = Y.shape[0]

        # Initialize the one-hot matrix with zeros
        one_hot_matrix = np.zeros((classes, m))

        # Set the appropriate elements to 1
        one_hot_matrix[Y, np.arange(m)] = 1

        return one_hot_matrix
    except Exception:
        return None
