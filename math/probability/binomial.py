#!/usr/bin/env python3
'''
Binomial distribution class.
'''


class Binomial:
    '''
    Binomial distribution class.
    '''

    def __init__(self, data=None, n=1, p=0.5):
        '''
        Initializes the binomial distribution object.

        Parameters:
        - data (list): A list of data points to calculate n and p from.
        - n (int): The number of trials.
        - p (float): The probability of success in each trial.
        '''
        if data is None:
            if n < 1:
                raise ValueError("n must be a positive value")
            else:
                self.n = int(n)
            if p <= 0 or p >= 1:
                raise ValueError("p must be greater than 0 and less than 1")
            else:
                self.p = float(p)
        else:
            if not isinstance(data, list):
                raise TypeError('data must be a list')
            if len(data) < 2:
                raise ValueError('data must contain multiple values')
            mean = float(sum(data) / len(data))
            variance = sum((x - mean) ** 2 for x in data) / len(data)
            q = variance / mean
            p = 1 - q
            n = round(mean / p)
            p = mean / n
            self.n = int(n)
            self.p = p

    def pmf(self, k):
        '''
        Calculates the value of the PMF for a given
        number of successes.

        Parameters:
        - k (int): The number of successes.

        Returns:
        - float: The probability mass function (PMF) value for
        the given number of successes.
        '''
        k = int(k)
        if k < 0:
            return 0
        binomial_co = self._binomial_coefficient(self.n, k)
        pmf = binomial_co * (self.p ** k) * ((1 - self.p) ** (self.n - k))
        return pmf

    def cdf(self, k):
        '''
        Calculates the value of the CDF for a given
        number of successes.

        Parameters:
        - k (int): The number of successes.

        Returns:
        - float: The cumulative distribution function
        (CDF) value for the given number of successes.
        '''
        k = int(k)
        if k < 0:
            return 0
        cdf = sum(self.pmf(i) for i in range(k + 1))
        return cdf

    @staticmethod
    def _binomial_coefficient(n, k):
        '''
        Calculates the binomial coefficient.

        Parameters:
        - n (int): Total number of trials.
        - k (int): Number of successes.

        Returns:
        - int: The binomial coefficient.
        '''
        if k == 0:
            return 1
        if k < 0 or k > n:
            return 0
        numerator = 1
        for i in range(n - k + 1, n + 1):
            numerator *= i
        denominator = 1
        for i in range(1, k + 1):
            denominator *= i
        return numerator // denominator
