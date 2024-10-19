#!/usr/bin/env python3
"""Performs PCA on a dataset using SVD."""


import numpy as np


def pca(X, var=0.95):
    """
    Performs PCA on the given dataset to
    retain the specified fraction of variance.

    Parameters:
        X (numpy.ndarray): Shape (n, d), dataset with n samples and d features.
        var (float): Fraction of variance to retain (default is 0.95).

    Returns:
        W (numpy.ndarray): Weight matrix of shape (d, nd) that projects X
                           to the new nd-dimensional space while
                           maintaining the specified variance.
    """
    # Perform SVD: X = U * S * V.T
    _, s, Vt = np.linalg.svd(X, full_matrices=False)

    # Calculate the cumulative variance explained by the singular values
    variance_ratios = np.cumsum(s) / np.sum(s)

    # Find the number of components to retain the desired variance
    nd = np.searchsorted(variance_ratios, var) + 1

    # Extract the top nd principal components (weight matrix)
    W = Vt.T[:, :nd]

    return W
