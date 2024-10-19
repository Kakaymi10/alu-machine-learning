#!/usr/bin/env python3
"""Performs PCA on a dataset using SVD with mean centering."""


import numpy as np


def pca(X, ndim):
    """
    Performs PCA on the given dataset to reduce it to ndim dimensions.

    Parameters:
        X (numpy.ndarray): Shape (n, d), dataset with n samples and d features.
        ndim (int): New dimensionality of the dataset.

    Returns:
        T (numpy.ndarray): Shape (n, ndim), the transformed dataset.
    """
    # Step 1: Center the data by subtracting the mean
    mean = np.mean(X, axis=0, keepdims=True)
    A = X - mean  # A is the mean-centered data

    # Step 2: Perform SVD: A = U * S * V.T
    _, _, Vt = np.linalg.svd(A, full_matrices=False)

    # Step 3: Extract the top ndim principal components (weight matrix)
    W = Vt.T[:, :ndim]

    # Step 4: Project the mean-centered data onto the new space
    T = np.matmul(A, W)

    return T
