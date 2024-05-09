#!/usr/bin/env python3
"""
Neural Class
"""


import numpy as np


class Neuron:
    """
    Neuron class represents a single neuron in a neural network.

    Attributes:
        nx (int): The number of input features.
        W (ndarray): The weights vector for the neuron.
        b (float): The bias for the neuron.
        A (float): The activated output of the neuron (forward propagation).
    """

    def __init__(self, nx):
        """
        Initializes a neuron.

        Args:
            nx (int): The number of input features.

        Raises:
            TypeError: If nx is not an integer.
            ValueError: If nx is not a positive integer.
        """
        if type(nx) != int:
            raise TypeError('nx must be an integer')
        if nx < 1:
            raise ValueError('nx must be a positive integer')
        self.nx = nx
        self.W = np.random.randn(1, nx)
        self.b = 0
        self.A = 0
