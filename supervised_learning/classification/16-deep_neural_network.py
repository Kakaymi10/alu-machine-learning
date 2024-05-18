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
        if not all(isinstance(i, int) and i > 0 for i in layers):
            raise TypeError('layers must be a list of positive integers')
        
        self.nx = nx
        self.layers = layers
        self.L = len(layers)  # Number of layers
        self.cache = {}  # Dictionary to store the values of the nodes
        self.weights = {}  # Dictionary to store the weights and biases

        # Initialize weights and biases
        for l in range(self.L):
            if l == 0:
                self.weights['W' + str(l + 1)] = np.random.randn(layers[l], nx) * np.sqrt(2. / nx)
            else:
                self.weights['W' + str(l + 1)] = np.random.randn(layers[l], layers[l - 1]) * np.sqrt(2. / layers[l - 1])
            self.weights['b' + str(l + 1)] = np.zeros((layers[l], 1))
