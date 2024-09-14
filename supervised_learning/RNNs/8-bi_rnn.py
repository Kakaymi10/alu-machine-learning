#!/usr/bin/env python3
'''
Bidirectional Cell Forward
'''


import numpy as np


def bi_rnn(bi_cell, X, h_0, h_t):
    '''
    Function that performs forward propagation for a bidirectional RNN
    '''
    t, m, i = X.shape
    h = h_0.shape[1]  # The dimensionality of the hidden state

    # Initialize forward and backward hidden state arrays
    Hf = np.zeros((t, m, h))  # Forward hidden states
    Hb = np.zeros((t, m, h))  # Backward hidden states

    # Set the initial hidden states
    Hf[0] = h_0  # Initialize forward hidden state
    Hb[-1] = h_t  # Initialize backward hidden state

    # Forward pass
    for step in range(1, t):
        Hf[step] = bi_cell.forward(Hf[step - 1], X[step])

    # Backward pass
    for step in range(t - 2, -1, -1):
        Hb[step] = bi_cell.backward(Hb[step + 1], X[step])

    # Concatenate forward and backward hidden states along the last axis
    H = np.concatenate((Hf, Hb), axis=-1)

    # Compute the output using the concatenated hidden states
    Y = bi_cell.output(H)  # Ensure bi_cell.output handles 2h input

    return H, Y
