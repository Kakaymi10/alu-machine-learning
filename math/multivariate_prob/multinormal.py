#!/usr/bin/env python3
"""
This module contains a class representing a
Multivariate Normal distribution.
"""

import numpy as np


class MultiNormal:
    """
    Represents a Multivariate Normal distribution.
    """
    def __init__(self, data):
        """
        Initialize the MultiNormal instance.

        Args:
            data (numpy.ndarray): The input dataset of shape (d, n),
            where n is the number of data points
                                  and d is the number
                                  of dimensions in each data point.
        """
        if not isinstance(data, np.ndarray) or data.ndim != 2:
            raise TypeError("data must be a 2D numpy.ndarray")
        d, n = data.shape
        if n < 2:
            raise ValueError("data must contain multiple data points")

        self.mean = np.mean(data, axis=1, keepdims=True)
        centered_data = data - self.mean
        self.cov = np.dot(centered_data, centered_data.T) / (n - 1)

    def pdf(self, x):
        """ calculate a PDF"""
        if type(x) is not np.ndarray:
            raise TypeError("x must be a numpy.ndarray")
        d = self.cov.shape[0]
        if len(x.shape) != 2 or x.shape != (d, 1):
            raise ValueError('x must have the shape ({}, 1)'.format(d))
        m = self.mean
        cov = self.cov
        bottom = np.sqrt(((2 * np.pi) ** d) * (np.linalg.det(cov)))
        inv = np.linalg.inv(cov)
        exp = (-.5 * np.matmul(np.matmul((x - m).T, inv), (x - m)))
        result = (1 / bottom) * np.exp(exp[0][0])
        return result
