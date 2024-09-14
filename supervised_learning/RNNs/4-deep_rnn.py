#!/usr/bin/env python3
'''
Deep RNN
'''

import numpy as np


def deep_rnn(rnn_cells, X, h_0):
    '''
    Function that performs forward propagation for a deep RNN
    '''
    t, m, i = X.shape
    l, _, h = rnn_cells
    H = np.zeros((t + 1, l, m, h))
    Y = np.zeros((t, m, rnn_cells[-1][-1]))
    H[0] = h_0
    for i in range(t):
        for j in range(l):
            if j == 0:
                H[i + 1, j], Y[i] = rnn_cells[j].forward(H[i, j], X[i])
            else:
                H[i + 1, j], Y[i] = rnn_cells[j].forward(H[i + 1, j - 1], H[i, j], X[i])
    return H, Y
