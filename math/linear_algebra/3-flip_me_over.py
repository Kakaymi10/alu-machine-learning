#!/usr/bin/env python3
'''
Write a function def matrix_transpose(matrix):
that returns the transpose of a 2D matrix, matrix:
You must return a new matrix
You can assume that matrix is never empty
You can assume all elements in the same
dimension are of the same type/shape
'''


def matrix_transpose(matrix):
    '''
    This function computes the row ans columns
    and return a transposition of a 2D matrix
    '''
    transposed = [[] for _ in range(len(matrix[0]))]
    for row in matrix:
        for col in range(len(row)):
            transposed[col].append(row[col])
    return transposed
