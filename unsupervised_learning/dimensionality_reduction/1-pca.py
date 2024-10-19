#!/usr/bin/env python3
"""Performs PCA on a dataset using SVD."""


import numpy as np


def pca(X, ndim):
    """
    Performs PCA on the given dataset to reduce it to ndim dimensions.

    Parameters:
        X (numpy.ndarray): Shape (n, d), dataset with n samples and d features.
        ndim (int): New dimensionality of the dataset.

    Returns:
        T (numpy.ndarray): Shape (n, ndim), the new dataset.
    """
    # Perform SVD: X = U * S * V.T
    _, _, Vt = np.linalg.svd(X, full_matrices=False)

    # Extract the top ndim principal components (weight matrix)
    W = Vt.T[:, :ndim]

    # Project X to the new space
    T = np.matmul(X, W)

    return (T)