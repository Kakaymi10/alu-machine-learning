#!/usr/bin/*env python3
"""
Poisson s lambtha
"""


class Poisson:
    def __init__(self, data=None, lambtha=1.):
        # Check if data is provided
        if data is None:
            # If not, use the provided lambtha
            self.lambtha = float(lambtha)
        else:
            # If data is provided, calculate lambtha from data
            if not isinstance(data, list):
                raise TypeError("data must be a list")
            if len(data) < 2:
                raise ValueError("data must contain multiple values")
            self.lambtha = sum(data) / len(data)

        # Check if lambtha is positive
        if self.lambtha <= 0:
            raise ValueError("lambtha must be a positive value")
