#!/usr/bin/env python3
'''
Bidirectional Cell Forward
'''


import numpy as np


class BidirectionalCell:
    '''
    Class that represents a bidirectional cell of an RNN
    '''
    def __init__(self, i, h, o):
        '''
        Class constructor
        '''
        self.Whf = np.random.normal(size=(h + i, h))
        self.Whb = np.random.normal(size=(h + i, h))
        self.Wy = np.random.normal(size=(2 * h, o))
        self.bhf = np.zeros((1, h))
        self.bhb = np.zeros((1, h))
        self.by = np.zeros((1, o))

    def forward(self, h_prev, x_t):
        ''' Method that performs forward propagation for one time step '''
        h_x = np.hstack((h_prev, x_t))
        h_next = np.tanh(np.dot(h_x, self.Whf) + self.bhf)
        return h_next

    def backward(self, h_next, x_t):
        ''' Method that performs backward propagation for one time step '''
        h_x = np.hstack((h_next, x_t))
        h_prev = np.tanh(np.dot(h_x, self.Whb) + self.bhb)
        return h_prev
