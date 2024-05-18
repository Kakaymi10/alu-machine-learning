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
        # first forward
        Z1 = np.dot(self.W1, X) + self.b1
        sigmoid1 = 1 / (1 + np.exp(-Z1))
        self.__A1 = sigmoid1

        # second forward
        Z2 = np.dot(self.W2, self.A1) + self.b2
        sigmoid2 = 1 / (1 + np.exp(-Z2))
        self.__A2 = sigmoid2

        return self.__A1, self.__A2

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
        s, class_prediction = self.forward_prop(X)
        cost = self.cost(Y, class_prediction)
        # Labelize the predictions:
        # if prediction < 0.5, set to 0; else, set to 1
        labelized = np.where(class_prediction < 0.5, 0, 1)
        return labelized, cost

    def gradient_descent(self, X, Y, A1, A2, alpha=0.05):
        """
        Perform one pass of gradient descent.
        X: input data
        Y: true labels
        A1: output of the hidden layer
        A2: predicted output
        alpha: learning rate
        """
        m = X.shape[1]
        # Output layer gradients
        dZ2 = A2 - Y
        dW2 = np.matmul(dZ2, A1.T) / m
        db2 = np.sum(dZ2, axis=1, keepdims=True) / m

        # Hidden layer gradients
        dA1 = np.matmul(self.__W2.T, dZ2)
        dZ1 = dA1 * A1 * (1 - A1)
        dW1 = np.matmul(dZ1, X.T) / m
        db1 = np.sum(dZ1, axis=1, keepdims=True) / m

        # Update weights and biases
        self.__W2 -= alpha * dW2
        self.__b2 -= alpha * db2
        self.__W1 -= alpha * dW1
        self.__b1 -= alpha * db1
