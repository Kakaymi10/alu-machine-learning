#!/usr/bin/env python3
"""
A function `add_matrices(mat1, mat2)`
that adds two matrices.
"""

def add_matrices(mat1, mat2):
    """
    Adds two matrices.
    Parameters:
    mat1 (list or int or float): The first matrix or element.
    mat2 (list or int or float): The second matrix or element.
    Returns:
    list or int or float: The result
    of matrix addition or the sum of two elements.
    Returns None if matrices are not compatible for addition.
    """
    if isinstance(mat1, list) and isinstance(mat2, list):
        if len(mat1) == len(mat2):
            return [add_matrices(a, b) for a, b in zip(mat1, mat2)]
    elif isinstance(mat1, (int, float)) and isinstance(mat2, (int, float)):
        return mat1 + mat2
    return None
