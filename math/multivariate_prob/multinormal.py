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
        """
        Calculate the Probability Density Function (PDF) at a data point.

        Args:
            x (numpy.ndarray): The data point of shape (d, 1), where
            d is the number of dimensions.

        Returns:
            float: The value of the PDF.
        """
        if not isinstance(x, np.ndarray):
            raise TypeError("x must be a numpy.ndarray")
        if x.shape != (self.mean.shape[0], 1):
            raise ValueError("x must have the shape ({}, 1)"
                             .format(self.mean.shape[0]))

        d = self.mean.shape[0]
        normalization = 1 / np.sqrt((2 * np.pi) ** d * np.linalg.det(self.cov))
        a = np.linalg.inv(self.cov))
        b = (x - self.mean)
        exponent = -0.5 * np.dot(np.dot((x - self.mean).T, a, b)

        pdf_value = normalization * np.exp(exponent)

        return pdf_value
