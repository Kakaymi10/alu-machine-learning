#!/usr/bin/env python3

import numpy as np

def pca(X, var=0.95):
    '''
    Performs PCA on a dataset.
    
    Parameters:
    - X: numpy.ndarray of shape (n, d) where:
      n is the number of data points
      d is the number of dimensions in each point
      All dimensions have a mean of 0 across data points
    - var: float, the fraction of variance to maintain (default: 0.95)
    
    Returns:
    - W: numpy.ndarray of shape (d, nd), the projection matrix
         where nd is the new dimensionality of the transformed X.
    '''
    # Step 1: Compute the covariance matrix of X
    cov = np.cov(X, rowvar=False)

    # Step 2: Compute eigenvalues and eigenvectors
    w, v = np.linalg.eig(cov)

    # Step 3: Sort eigenvalues and eigenvectors in descending order
    idx = np.argsort(w)[::-1]
    w = w[idx]
    v = v[:, idx]

    # Step 4: Compute cumulative explained variance
    cum_exp_var = np.cumsum(w) / np.sum(w)

    # Step 5: Find the minimum number of dimensions needed to retain the desired variance
    d = np.argmax(cum_exp_var >= var) + 1

    # Step 6: Create the projection matrix W using the top 'd' eigenvectors
    W = v[:, :d]
    
    return W
