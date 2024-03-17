#!/usr/bin/env python3
"""
Write a function def likelihood(x, n, P): that calculates
the likelihood of obtaining this data given various
hypothetical probabilities of developing severe side effects:
"""

import numpy as np


def likelihood(x, n, P):
    """
    Calculates the likelihood of obtaining data given various hypothetical
    probabilities of developing severe side effects.

    Args:
        x (int): Number of patients that develop severe side effects.
        n (int): Total number of patients observed.
        P (numpy.ndarray): 1D array containing various hypothetical
        probabilities
        of developing severe side effects.

    Raises:
        ValueError: If n is not a positive integer, or x is not an
        integer >= 0, or
        x is greater than n, or any value in P is not in the range [0, 1].
        TypeError: If P is not a 1D numpy.ndarray.

    Returns:
        numpy.ndarray: Likelihood of obtaining the data, x
        and n, for each probability in P.
    """
    if not isinstance(n, int) or n <= 0:
        raise ValueError("n must be a positive integer")
    if not isinstance(x, int) or x < 0:
        text = "x must be an integer that is greater than or equal to 0"
        raise ValueError(text)
    if x > n:
        raise ValueError("x cannot be greater than n")
    if not isinstance(P, np.ndarray) or P.ndim != 1:
        raise TypeError("P must be a 1D numpy.ndarray")
    if np.any((P < 0) | (P > 1)):
        raise ValueError("All values in P must be in the range [0, 1]")

    denom = (np.math.factorial(x) * np.math.factorial(n - x))
    binomial_coeff = np.math.factorial(n) / denom
    likelihoods = binomial_coeff * np.power(P, x) * np.power(1 - P, n - x)

    return likelihoods
