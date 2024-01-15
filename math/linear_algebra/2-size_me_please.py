#!/usr/bin/env/python3
'''Write a function def matrix_shape(matrix): that calculates the shape of a matrix:

You can assume all elements in the same dimension are of the same type/shape
The shape should be returned as a list of integers '''

def matrix_shape(matrix):
    rows = matrix
    result = []
    while len(rows) > 0:
        result.append(len(rows))
        if isinstance(rows[0], int):
            break
        rows = rows[0]
    return result
