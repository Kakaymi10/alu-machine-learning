#!/usr/bin/env python3
'''
Write a function def np_matmul(mat1, mat2): that performs matrix multiplication
'''

import numpy as np


def np_matmul(mat1, mat2):
    '''
    This function multiplies two matrices
    np.matmul(matrix1, matrix2)
    np.dot(matrix1, matrix2)
    '''
    return mat1@mat2
