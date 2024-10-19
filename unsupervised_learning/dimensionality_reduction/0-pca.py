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
    # Compute covariance matrix
    cov = np.cov(X, rowvar=False)  # rowvar=False ensures columns are variables

    # Get eigenvalues (w) and eigenvectors (v)
    w, v = np.linalg.eig(cov)

    # Sort eigenvalues and eigenvectors in descending order
    idx = np.argsort(w)[::-1]
    w = w[idx]
    v = v[:, idx]

    # Calculate cumulative explained variance
    cum_exp_var = np.cumsum(w) / np.sum(w)

    # Determine the number of dimensions needed to maintain 'var' variance
    d = np.argmax(cum_exp_var >= var) + 1

    # Create the projection matrix using the top 'd' eigenvectors
    W = v[:, :d]
    
    return W
