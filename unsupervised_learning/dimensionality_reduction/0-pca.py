#!/usr/bin/env python3


import numpy as np

def pca(X, var=0.95):
    """
    Performs PCA on the dataset X to maintain the given fraction of variance.

    Parameters:
    X (numpy.ndarray): Shape (n, d), where n is the number of samples, 
                       and d is the number of features (dimensions).
    var (float): Fraction of variance to maintain (default is 0.95).

    Returns:
    numpy.ndarray: The weight matrix W of shape (d, nd), where nd is the 
                   new dimensionality that preserves the desired variance.
    """
    # Step 1: Compute the covariance matrix of X
    cov_matrix = np.cov(X, rowvar=False)

    # Step 2: Perform eigen decomposition of the covariance matrix
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

    # Step 3: Sort eigenvalues and corresponding eigenvectors in descending order
    sorted_indices = np.argsort(eigenvalues)[::-1]
    sorted_eigenvalues = eigenvalues[sorted_indices]
    sorted_eigenvectors = eigenvectors[:, sorted_indices]

    # Step 4: Calculate the cumulative variance explained
    cumulative_variance = np.cumsum(sorted_eigenvalues) / np.sum(sorted_eigenvalues)

    # Step 5: Find the number of components to retain the desired variance
    nd = np.argmax(cumulative_variance >= var) + 1

    # Step 6: Select the top nd eigenvectors (the weight matrix W)
    W = sorted_eigenvectors[:nd]

    return W
