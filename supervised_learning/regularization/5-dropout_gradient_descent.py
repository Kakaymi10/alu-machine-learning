#!/usr/bin/env python3
""" Gradient Descent with Dropout """


import numpy as np


def dropout_gradient_descent(Y, weights, cache, alpha, keep_prob, L):
    """
    Update the weights of a neural network using gradient descent with dropout regularization.
    """
    m = Y.shape[1]
    dz = cache[f'A{L}'] - Y
    for i in range(L, 0, -1):
        A_prev = cache[f'A{i - 1}']
        W = weights[f'W{i}']
        dW = np.matmul(dz, A_prev.T) / m
        db = np.sum(dz, axis=1, keepdims=True) / m

        if i > 1:
            dz = np.matmul(W.T, dz) * (1 - np.square(A_prev)) * cache[f'D{i - 1}']
            dz = dz / keep_prob

        weights[f'W{i}'] -= alpha * dW
        weights[f'b{i}'] -= alpha * db
