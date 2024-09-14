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
    h = h_0.shape[2]
    Hf = np.zeros((t, m, h))
    Hb = np.zeros((t, m, h))
    Hf[0] = h_0
    Hb[-1] = h_t
    for i in range(t):
        Hf[i] = bi_cell.forward(Hf[i - 1], X[i])
    for i in range(t - 1, -1, -1):
        Hb[i] = bi_cell.backward(Hb[i + 1], X[i])
    H = np.hstack((Hf, Hb))
    Y = bi_cell.output(H)
    return H, Y
