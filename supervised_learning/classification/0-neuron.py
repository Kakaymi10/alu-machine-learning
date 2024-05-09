#!/usr/bin/env python3

import numpy as np

class Neuron:
    def __init__(self, nx):
        if type(nx) != int:
            raise TypeError('nx must be a integer')
        if nx < 0:
            raise ValueError('nx must be positive')
        self.nx = nx
        self.W = np.random.randn(size=nx)
        self.b = 0
        self.A = 0
