#!/usr/bin/env python3
'''
0. PCA
'''


import numpy as np


def pca(X, var=0.95):
    '''
    Function that performs PCA on a dataset
    '''
    
    # Covariance matrix
    cov = np.cov(X.T)
    # Eigenvalues and eigenvectors
    w, v = np.linalg.eig(cov)
    # Sort eigenvalues
    w_sort = np.sort(w)[::-1]
    # Explained variance
    exp_var = w_sort / np.sum(w)
    # Cumulative explained variance
    cum_exp_var = np.cumsum(exp_var)
    # Number of dimensions to keep
    d = np.argwhere(cum_exp_var >= var)[0, 0]
    # Projection matrix
    W = v[:, :d + 1]
    return W