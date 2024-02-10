
#!/usr/bin/env python3
"""
Poisson s lambtha
"""


class Poisson:
    """
    Poisson class
    """
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

    def pmf(self, k):
        k = int(k)
        if k < 0:
            return 0  # If k is negative, return 0
        else:
            return math.exp(-self.lambtha) * (self.lambtha ** k) / math.factorial(k)
