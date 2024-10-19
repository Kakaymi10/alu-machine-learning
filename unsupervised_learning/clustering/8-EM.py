#!/usr/bin/env python3
"""Finding the best number of clusters for a GMM using the Bayesian Information Criterion"""

import numpy as np
# Import the necessary functions from their respective modules
initialize = __import__('4-initialize').initialize
expectation = __import__('6-expectation').expectation
maximization = __import__('7-maximization').maximization

def expectation_maximization(X, k, iterations=1000, tol=1e-5, verbose=False):
    """
    Performs the Expectation-Maximization (EM) algorithm for a Gaussian Mixture Model (GMM).
    
    Parameters:
    - X: numpy.ndarray of shape (n, d) containing the dataset
    - k: positive integer, number of clusters
    - iterations: positive integer, maximum number of iterations
    - tol: non-negative float, tolerance for log likelihood for early stopping
    - verbose: boolean, if True prints log likelihood every 10 iterations and at the end
    
    Returns:
    - pi: numpy.ndarray of shape (k,) containing the priors for each cluster
    - m: numpy.ndarray of shape (k, d) containing the centroid means for each cluster
    - S: numpy.ndarray of shape (k, d, d) containing the covariance matrices for each cluster
    - g: numpy.ndarray of shape (k, n) containing the probabilities for each data point in each cluster
    - l: log likelihood of the model
    """
    n, d = X.shape
    
    # Initialize parameters
    pi, m, S = initialize(X, k)
    
    l = 0  # Initial log likelihood
    for i in range(iterations):
        # E-step
        g, l_new = expectation(X, pi, m, S)
        
        # M-step
        pi, m, S = maximization(X, g)
        
        # Check for convergence
        if abs(l_new - l) <= tol:
            break
        
        l = l_new
        
        # Verbose logging
        if verbose and i % 10 == 0:
            print(f"Log Likelihood after {i} iterations: {l:.5f}")
    
    # Final log likelihood print
    if verbose:
        print(f"Log Likelihood after {i} iterations: {l:.5f}")

    return (pi, m, S, g, l) if l > -np.inf else (None, None, None, None, None)