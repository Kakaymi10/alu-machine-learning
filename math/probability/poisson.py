#!/usr/bin/env python3
"""
Poisson s lambtha
"""


class Poisson:
    """
    Poisson class
    """
    def __init__(self, data=None, lambtha=1.):
        '''
        Check if data is provided
        '''
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
        self.expo = 2.7182818285
        self.pi = 3.1415926536

    def factorial(self, k):
        '''
        Calculates the factorial of a number for the pmf
        '''
        if k < 0:
            return 0
        if k == 0 or k == 1:
            return 1
        return k * self.factorial(k - 1)

    def pmf(self, k):
        '''
        calculates the probabilti mass function of a poison dist
        '''
        k = int(k)
        if k < 0:
            return 0  # If k is negative, return 0
        else:
            numerator = (self.expo**(-self.lambtha) * (self.lambtha ** k))
            return numerator / self.factorial(k)

    def cdf(self, k):
        '''
        calcuates the cdf of a poison distrubtion
        '''
        k = int(k)
        if k < 0:
            return 0
        cdf = 0
        for i in range(k+1):
            cdf += self.pmf(i)
        return cdf
