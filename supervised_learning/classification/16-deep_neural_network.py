#!/usr/bin/env python3
"""
Deep Neural Network Class
"""


import numpy as np


class DeepNeuralNetwork:
    """
    A class that  represents a deep neural network.
    """
    def __init__(self, nx, layers):
        if type(nx) != int:
            raise TypeError('nx must be an integer')
        if nx < 1:
            raise ValueError('nx must be a positive integer')
        if type(layers) != list:
            raise TypeError('layers must be a list of positive integers')

        self.nx = nx
        self.layers = []
        self.L = len(layers)  # Number of layers
        self.cache = {}  # Dictionary to store the values of the nodes
        self.weights = {}  # Dictionary to store the weights and biases

        # Initialize weights and biases
        for l in range(self.L):
            if not isinstance(layers[l], int) or layers[l] <= 0:
                raise TypeError('layers must be a list of positive integers')

            l_inp_s = nx if l == 0 else layers[l - 1]
            weight_key = 'W' + str(l + 1)
            self.weights[weight_key] = np.random.randn(layers[l], l_inp_s) * np.sqrt(2. / l_inp_s)
            self.weights['b' + str(l + 1)] = np.zeros((layers[l], 1))
