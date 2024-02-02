#!/usr/bin/env python3
"""
summs
"""


def summation_i_squared(n):
    """Calculate the sum of squares of the first n natural numbers.

    Args:
        n (int): The number of natural numbers to consider.

    Returns:
        int or None: The sum of squares if n is a positive integer, else None.
    """
    return (n * (n + 1) * (2 * n + 1)) // 6 if n > 0 else None
