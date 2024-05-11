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
        self.__W = np.random.normal(size=(1, nx))
        self.__b = 0
        self.__A = 0

    @property
    def W(self):
        return self.__W

    @W.setter
    def W(self, value):
        self.__W = value

    @property
    def b(self):
        return self.__b

    @b.setter
    def b(self, value):
        self.__b = value

    @property
    def A(self):
        return self.__A

    def forward_prop(self, X):
        """
        Forward propagates the input data through the neuron.

        Arguments:
        X: numpy.ndarray - Input data with shape (nx, m).

        Returns:
        numpy.ndarray - The activated output of the neuron.
        """
        Z = np.dot(self.W, X) + self.b
        sigmoid = 1 / (1 + np.exp(-Z))
        self.__A = sigmoid
        return self.__A

    def cost(self, Y, A):
        """
        Calculate the cross-entropy loss function.

        Parameters:
        Y (numpy array): Ground truth labels, shape (1, m).
        A (numpy array): Predicted probabilities, shape (1, m).

        Returns:
        float: Cross-entropy loss.
        """
        log_loss_arr = -(Y)*np.log(A) - (1-Y)*np.log(1.0000001-A)
        sum = np.sum(log_loss_arr)
        length = log_loss_arr.size
        cost = sum / length
        return cost

    def evaluate(self, X, Y):
        """
        Evaluate the model's predictions and cost on given input data.

        Parameters:
        X (numpy array): Input data, shape (input_size, m).
        Y (numpy array): Ground truth labels, shape (1, m).

        Returns:
        str: A formatted string containing labelized predictions and cost.
        """
        class_prediction = self.forward_prop(X)
        cost = self.cost(Y, class_prediction)
        # Labelize the predictions:
        # if prediction < 0.5, set to 0; else, set to 1
        labelized = np.where(class_prediction < 0.5, 0, 1)
        return (labelized, cost)
