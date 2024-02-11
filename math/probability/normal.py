#!/usr/bin/env python3
'''
This module provides functionality related to the normal distribution.
'''


class NormalDistribution:
    '''
    A class representing a normal (Gaussian) distribution.
    '''

    def __init__(self, data=None, mean=0., stddev=1.):
        '''
        Initializes the normal distribution object.

        Parameters:
        - data (list): A list of data points to calculate mean
        and standard deviation from.
        - mean (float): Mean of the distribution.
        - stddev (float): Standard deviation of the distribution.
        '''

        if data is None:
            if stddev < 1:
                raise ValueError("stddev must be a positive value")
            else:
                self.stddev = float(stddev)
                self.mean = float(mean)
        else:
            if not isinstance(data, list):
                raise TypeError("data must be a list")
            elif len(data) < 2:
                raise ValueError("data must contain multiple values")
            else:
                mean = float(sum(data) / len(data))
                self.mean = mean
                summation = sum((x - mean) ** 2 for x in data)
                stddev = (summation / len(data)) ** 0.5
                self.stddev = stddev
        self.e = 2.7182818285
        self.pi = 3.1415926536

    def z_score(self, x):
        '''
        Calculates the z-score of a given x-value.

        Parameters:
        - x (float): The value for which z-score is to be calculated.

        Returns:
        - float: The z-score of the given x-value.
        '''
        return (x - self.mean) / self.stddev

    def x_value(self, z):
        '''
        Calculates the x-value of a given z-score.
        Parameters:
        - z (float): The z-score for which x-value is to be calculated.
        Returns:
        - float: The x-value corresponding to the given z-score.
        '''
        return (z * self.stddev) + self.mean

    def pdf(self, x):
        '''
        Calculates the value of the PDF for a given x-value.
        Parameters:
        - x (float): The value for which PDF is to be calculated.
        Returns:
        - float: The probability density function (PDF) value for the given x-value.
        '''
        power = -0.5 * (self.z_score(x) ** 2)
        coefficient = 1 / (self.stddev * ((2 * self.pi) ** 0.5))
        pdf = coefficient * (self.e ** power)
        return pdf

    def cdf(self, x):
        '''
        Calculates the value of the CDF for a given x-value.
        Parameters:
        - x (float): The value for which CDF is to be calculated.
        Returns:
        - float: The cumulative distribution function (CDF) value for the given x-value.
        '''
        value = (x - self.mean) / (self.stddev * (2 ** 0.5))
        val = value - ((value ** 3) / 3) + ((value ** 5) / 10) - ((value ** 7) / 42) + ((value ** 9) / 216)
        val *= (2 / (self.pi ** 0.5))
        cdf = 0.5 * (1 + val)
        return cdf
