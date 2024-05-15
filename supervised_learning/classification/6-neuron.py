#!/usr/bin/env python3
"""
Neural Class
"""


import numpy as np
import matplotlib.pyplot as plt


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

    @property
    def b(self):
        return self.__b

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

    def gradient_descent(self, X, Y, A, alpha=0.05):
        """
        Perform one pass of gradient descent on the neuron.
        Arguments:
        - X: numpy.ndarray with shape (nx, m) containing the input data.
        - Y: numpy.ndarray with shape (1, m) containing the correct labels.
        - A: numpy.ndarray with shape (1, m) containing the activated output
        of the neuron for each example.
        - alpha: float representing the learning rate (default is 0.05).
        """
        diff = A - Y
        dcost_dw1 = np.dot(X, diff.T) / np.size(A)
        dcost_db1 = np.sum(diff) / np.size(A)
        self.__W -= (alpha)*(dcost_dw1.T)
        self.__b -= (alpha)*(dcost_db1)

    def train(self, X, Y, iterations=5000, alpha=0.05, verbose=True, graph=True, step=100):
        """
        Trains the neuron using gradient descent.

        Args:
            X (numpy.ndarray): Input data with shape (nx, m).
            Y (numpy.ndarray): Correct labels with shape (1, m).
            iterations (int): Number of iterations for
            training (default is 5000).
            alpha (float): Learning rate (default is 0.05).

        Returns:
            tuple: A tuple containing labelized predictions and cost.
        """
        # Check if iterations is an integer and positive
        if type(iterations) != int:
            raise TypeError('iterations must be an integer')
        if iterations <= 0:
            raise ValueError('iterations must be a positive integer')

        # Check if alpha is a float and positive
        if type(alpha) != float:
            raise TypeError('alpha must be a float')
        if alpha <= 0:
            raise ValueError('alpha must be positive')

        # Check if step is an integer, positive, and less than or equal to iterations
        if not isinstance(step, int):
            raise TypeError("step must be an integer")
        if step <= 0 or step > iterations:
            raise ValueError("step must be positive and <= iterations")
            
        costs = []
        # Iterate over the specified number of iterations
        for iteration in range(iterations):
            # Forward propagate to calculate the activated output
            A = self.forward_prop(X)

            # Calculate cost and store for plotting
            cost = self.cost(Y, A)
            costs.append(cost)
     
            self.gradient_descent(X, Y, A, alpha)

            # Print verbose information every 'step' iterations
            if verbose and iteration % step == 0:
                print(f"Cost after {iteration} iterations: {cost}")

        # Plot training cost if graph is True
        if graph:
            plt.plot(range(0, iterations + 1, step), costs, 'b-')
            plt.xlabel('Iteration')
            plt.ylabel('Cost')
            plt.title('Training Cost')
            plt.show()

        # Evaluate the model on the training data
        return self.evaluate(X, Y)
