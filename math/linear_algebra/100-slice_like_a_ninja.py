#!/usr/bin/env python3
"""
A function `np_slice(matrix, axes)` that slices
the matrix along specified axes.
"""

def np_slice(matrix, axes):
    """
    A function `np_slice(matrix, axes)`
    that slices the matrix along specified axes.
    """
    slices_matrix = [slice(None) for _ in range(len(matrix.shape))]

    for axis, value in axes.items():
        slices_matrix[axis] = slice(*value)

    return matrix[tuple(slices_matrix)]
