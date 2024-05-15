#!/usr/bin/env python3
"""
Neural Class
"""


import numpy as np


class NeuralNetwork:
    """
    A class that  represents a neural network.
    """

    def __init__(self, nx, nodes):
        """
        A class that  represents a neural network.
        """
        # Validate input parameters
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        if not isinstance(nodes, int):
            raise TypeError("nodes must be an integer")
        if nodes < 1:
            raise ValueError("nodes must be a positive integer")

        # Initialize weights and biases
        self.__W1 = np.random.randn(nodes, nx)
        self.__b1 = 0
        self.__A1 = 0
        self.__W2 = np.random.randn(1, nodes)
        self.__b2 = 0
        self.__A2 = 0

    @property
    def W1(self):
        return self.__W1

    @property
    def b1(self):
        return self.__b1

    @property
    def A1(self):
        return self.__A1

    @property
    def W2(self):
        return self.__W2

    @property
    def b2(self):
        return self.__b2

    @property
    def A2(self):
        return self.__A2

    def forward_prop(self, X):
        '''
        public method def forward_prop(self, X):
        Calculates the forward propagation of the
        neural network
        '''
        #first forward
        Z1 = np.dot(self.W1, X) + self.b1
        sigmoid1 = 1 / (1 + np.exp(-Z1))
        self.A1 = sigmoid1

        #second forward
        Z2 = np.dot(self.W2, self.A1) + self.b2
        sigmoid2 = 1 / (1 + np.exp(-Z2))
        self.A2 = sigmoid2

        return (self.A1, self.A2)
