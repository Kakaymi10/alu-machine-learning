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
    h = h_0.shape[1]

    Hf = np.zeros((t, m, h))
    Hb = np.zeros((t, m, h))

    Hf[0] = h_0
    Hb[-1] = h_t

    # Forward pass
    for i in range(1, t):
        Hf[i] = bi_cell.forward(Hf[i - 1], X[i])

    # Backward pass
    for i in range(t - 2, -1, -1):
        Hb[i] = bi_cell.backward(Hb[i + 1], X[i])

    # Concatenate forward and backward hidden states
    H = np.hstack((Hf, Hb))

    # Output computation
    Y = bi_cell.output(H)

    return H, Y
