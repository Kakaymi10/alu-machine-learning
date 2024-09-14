#!/usr/bin/env python3
'''
Deep RNN
'''


import numpy as np


def deep_rnn(rnn_cells, X, h_0):
    '''
    Function that performs forward propagation for a deep RNN
    '''
    num_layers = len(rnn_cells)
    t, m, i = X.shape
    h = h_0.shape[2]
    H = np.zeros((t + 1, num_layers, m, h))
    Y = []
    H[0] = h_0
    for i in range(t):
        h_prev = X[i]
        for j in range(num_layers):
            h_prev, y = rnn_cells[j].forward(H[i, j], h_prev)
            H[i + 1, j] = h_prev
        Y.append(y)
    Y = np.array(Y)
    return H, Y
