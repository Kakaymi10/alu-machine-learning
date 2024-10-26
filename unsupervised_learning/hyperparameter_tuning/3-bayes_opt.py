#!/usr/bin/env python3
"""
Bayesian optimization
"""

import numpy as np
GP = __import__('2-gp').GaussianProcess


class BayesianOptimization:
    """
    Bayesian optimization on a noiseless 1D Gaussian process
    """
    def __init__(self, f, X_init, Y_init, bounds, ac_samples, l=1, sigma_f=1,
                 xsi=0.01, minimize=True):
        """
        init method for bayesian optimization
        """
        # black-box function
        self.f = f

        # Gaussian Process
        self.gp = GP(X_init, Y_init, l, sigma_f)

        # X_s all acquisition sample
        X_s = np.linspace(bounds[0], bounds[1], num=ac_samples)
        self.X_s = X_s.reshape(-1, 1)

        # exploration-explotation
        self.xsi = xsi

        # minimization versus maximization
        self.minimize = minimize
