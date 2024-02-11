#!/usr/bin/env python3
'''
A class representing an exponential distribution.
'''


class Exponential:
    '''
    Class representing an exponential distribution.
    '''

    def __init__(self, data=None, lambtha=1.):
        '''
        Initializes the exponential distribution object.

        Parameters:
        - data (list): A list of data points to calculate lambda from.
        - lambtha (float): The rate parameter of the distribution.
        '''
        if data is None:
            if lambtha <= 0:
                raise ValueError('lambtha must be a positive value')
            self.lambtha = float(lambtha)
        else:
            if not isinstance(data, list):
                raise TypeError('data must be a list')
            if len(data) < 2:
                raise ValueError('data must contain multiple values')
            self.lambtha = float(1 / (sum(data) / len(data)))

    def pdf(self, x):
        '''
        Calculates the value of the PDF for a given time period.

        Parameters:
        - x (float): The time period for which PDF is to be calculated.

        Returns:
        - float: The probability density function (PDF) value
        for the given time period.
        '''
        if x < 0:
            return 0
        return self.lambtha * (2.7182818285 ** (-self.lambtha * x))

    def cdf(self, x):
        '''
        Calculates the value of the CDF for a given time period.
        Parameters:
        - x (float): The time period for which CDF is to be calculated.
        Returns:
        - float: The cumulative distribution function (CDF) value for
        the given time period.
        '''
        if x < 0:
            return 0
        return 1 - (2.7182818285 ** (-self.lambtha * x))
