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

    def backward(self, h_prev, x_t):
        ''' Method that performs backward propagation for one time step '''
        h_x = np.hstack((h_prev, x_t))  # Use h_prev and x_t, not h_next
        h_next = np.tanh(np.dot(h_x, self.Whb) + self.bhb)
        return h_next


    def output(self, H):
        ''' Method that calculates the output of the cell '''
        t, m, h = H.shape
        Y = np.zeros((t, m, self.Wy.shape[1]))

        for i in range(t):
            # No need to re-stack H, use it directly
            Y[i] = np.dot(H[i], self.Wy) + self.by
            exp_Y = np.exp(Y[i])
            Y[i] = exp_Y / np.sum(exp_Y, axis=1, keepdims=True)

        return Y
