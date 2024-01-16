#!/usr/bin/env python3
<<<<<<< HEAD
'''Write a function def matrix_shape(matrix): that calculates the shape of a matrix:
You can assume all elements in the same dimension are of the same type/shape
The shape should be returned as a list of integers '''
=======
'''
Write a function def matrix_shape(matrix):
that calculates the shape of a matrix:
You can assume all elements in the same
dimension are of the same type/shape
The shape should be returned as a list of integers
'''

>>>>>>> 18924bc4c444d0b61653d9a8a7b8e070d3adde80

def matrix_shape(matrix):
    '''
    this function computes the lengths and returns
    a shape
    '''
    rows = matrix
    shape = []
    while len(rows) > 0:
        shape.append(len(rows))
        rows = rows[0] if isinstance(rows[0], list) else []
    return shape
