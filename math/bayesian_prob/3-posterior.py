#!/usr/bin/env python3
"""Compute the marginal probability."""
import numpy as np


def posterior(x, n, P, Pr):
    """Return the posterior probability of obtaining x and n."""
    if not isinstance(n, int) or n <= 0:
        raise ValueError('n must be a positive integer')
    if not isinstance(x, int) or x < 0:
        text = 'x must be an integer that is greater than or equal to 0'
        raise ValueError(text)
    if x > n:
        raise ValueError('x cannot be greater than n')
    if not isinstance(P, np.ndarray) or len(P.shape) != 1:
        raise TypeError('P must be a 1D numpy.ndarray')
    if not all(0 <= i <= 1 for i in P):
        raise ValueError('All values in P must be in the range [0, 1]')
    if not isinstance(Pr, np.ndarray) or Pr.shape != P.shape:
        raise TypeError('Pr must be a numpy.ndarray with the same shape as P')
    if not all(0 <= i <= 1 for i in Pr):
        raise ValueError('All values in Pr must be in the range [0, 1]')
    if not np.isclose(np.sum(Pr), 1):
        raise ValueError('Pr must sum to 1')

    factorial = np.math.factorial
    likelihood = factorial(n) / (factorial(x) * factorial(n - x))
    likelihood *= (P ** x) * ((1 - P) ** (n - x))
    marginal = np.sum(likelihood * Pr)
    return likelihood * Pr / marginal
