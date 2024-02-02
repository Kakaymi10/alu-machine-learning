#!/usr/bin/env python3
"""
Calcul the integral of a polynomial
"""


def poly_integral(poly, C=0):
    """Calculate the integral of a polynomial.

    Args:
        poly (list): List of coefficients representing the polynomial.
        C (int): Constant of integration.

    Returns:
        list or None: List of coefficients representing
the integral of the polynomial.
        Returns None if input is not a non-empty list or C is not an integer.
    """
    # Check for the validity of the inputs
    if not isinstance(poly, list) or not poly or not isinstance(C, int):
        return None

    # Calculate the coefficients of the integral using the power rule
    integral = [poly[i] / (i + 1) for i in range(len(poly) - 1, 0, -1)]
    integral.append(poly[0] / 1)
    integral.append(C)

    # If the original polynomial is a constant term (0), set the intgl to [C]
    if len(poly) == 1 and poly[0] == 0:
        integral = [C]

    # Convert any floating-point coefficients to integers if they are ws
    integral = [int(coeff) if coeff % 1 == 0 else coeff for coeff in integral]

    return integral[::-1]
