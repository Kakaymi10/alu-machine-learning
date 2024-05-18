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
        if len(layers) < 1:
            raise TypeError('layers must be a list of positive integers')

        self.__L = len(layers)
        self.__cache = {}
        self.__weights = {}

        for i in range(self.__L):
            if not isinstance(layers[i], int) or layers[i] < 1:
                raise TypeError('layers must be a list of positive integers')

            if i == 0:
                # He et al. initialization
                self.__weights['W' + str(i + 1)] = np.random.randn(
                    layers[i], nx) * np.sqrt(2 / nx)
            else:
                # He et al. initialization
                self.__weights['W' + str(i + 1)] = np.random.randn(
                    layers[i], layers[i - 1]) * np.sqrt(2 / layers[i - 1])

            # Zero initialization
            self.__weights['b' + str(i + 1)] = np.zeros((layers[i], 1))

    @property
    def L(self):
        return self.__L

    @property
    def cache(self):
        return self.__cache

    @property
    def weights(self):
        return self.__weights

    def forward_prop(self, X):
        '''
        forward propagate deep neural network
        '''
        self.__cache['A0'] = X

        for layer in range(1, self.__L + 1):
            inpLayer = "A" + str(layer - 1)
            inpW = "W" + str(layer)
            inpBias = "b" + str(layer)
            wx = np.matmul(self.__weights[inpW], self.__cache[inpLayer])
            b = self.__weights[inpBias]
            z = wx + b
            sigmoid = 1 / (1 + np.exp(-z))
            outputLayer = 'A' + str(layer)
            self.__cache[outputLayer] = sigmoid

        return sigmoid, self.__cache

    def cost(self, Y, A):
        '''
        cost of deep neural network
        '''
        m = Y.shape[1]
        cost = -np.sum(Y * np.log(A) + (1 - Y) * np.log(1.0000001 - A)) / m
        return cost

    def evaluate(self, X, Y):
        """
        Evaluate the model's predictions and cost on given input data.
        """
        prediction, _ = self.forward_prop(X)
        output = self.cache.get("A" + str(self.L))
        cost = self.cost(Y, output)
        prediction = np.where(output >= 0.5, 1, 0)
        return prediction, cost
