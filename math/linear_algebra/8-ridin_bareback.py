#!/usr/bin/env python3
'''
Write a function def mat_mul(mat1, mat2):
that performs matrix multiplication:
You can assume that mat1 and mat2 are
2D matrices containing ints/floats
You can assume all elements in
the same dimension are of the same type/shape
You must return a new matrix
If the two matrices cannot be multiplied, return None
'''


def mat_mul(mat1, mat2):
    '''
    This function returns matrix multiplication
    '''
    if len(mat1[0]) != len(mat2):
        return None
    result = [[0 for _ in range(len(mat2[0]))] for _ in range(len(mat1))]
    for i in range(len(mat1)):
        for j in range(len(mat2[0])):
            for k in range(len(mat2)):
                result[i][j] += mat1[i][k] * mat2[k][j]
    return result
