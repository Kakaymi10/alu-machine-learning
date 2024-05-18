#!/usr/bin/env python3
"""
Deep Neural Network Class
"""


import numpy as np


def one_hot_decode(one_hot):
    '''One hot decode
    '''
    if not isinstance(one_hot, np.ndarray) or len(one_hot.shape) != 2:
        return None
    return np.argmax(one_hot, axis=0)
