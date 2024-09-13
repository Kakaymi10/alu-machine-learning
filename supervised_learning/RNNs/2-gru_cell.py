#!/usr/bin/env python3
'''
GRU Cell
'''


import numpy as np


class GRUCell:
    '''
    Class that represents a cell of a GRU
    '''
    def __init__(self, i, h, o):
        '''
        Class constructor
        '''
        self.Wz = np.random.normal(size=(h + i, h))
        self.Wr = np.random.normal(size=(h + i, h))
        self.Wh = np.random.normal(size=(h + i, h))
        self.Wy = np.random.normal(size=(h, o))
        self.bz = np.zeros((1, h))
        self.br = np.zeros((1, h))
        self.bh = np.zeros((1, h))
        self.by = np.zeros((1, o))

    def forward(self, h_prev, x_t):
        ''' Method that performs forward propagation for one time step '''
        h_x = np.hstack((h_prev, x_t))
        z = 1 / (1 + np.exp(-(np.dot(h_x, self.Wz) + self.bz)))
        r = 1 / (1 + np.exp(-(np.dot(h_x, self.Wr) + self.br)))
        h_x = np.hstack((r * h_prev, x_t))
        h_t = np.tanh(np.dot(h_x, self.Wh) + self.bh)
        h_next = (1 - z) * h_prev + z * h_t
        y = np.dot(h_next, self.Wy) + self.by
        y = np.exp(y) / np.sum(np.exp(y), axis=1, keepdims=True)
        return h_next, y
