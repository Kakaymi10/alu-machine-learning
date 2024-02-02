#!/usr/bin/env python3
"""
Calculates
"""


def poly_derivative(poly):
    """Find the derivative of a polynomial.

    Args:
    poly (list): List of coefficients representing the polynomial.

    Returns:
        list or None: List of coefficients representing
the derivative of the polynomial.
               Returns None if input is not a non-empty list.
    """
    # Check if poly is a non-empty list
    if not isinstance(poly, list) or not poly:
        return None

    # If poly has only one term, the derivative is a constant (0)
    if len(poly) == 1:
        return [0]

    # Calculate the derivative using the power rule
    derivative = [poly[i] * i for i in range(len(poly) - 1, 0, -1)]

    # Reverse the list to have the coefficients in the correct order
    return derivative[::-1]
