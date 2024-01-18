#!/usr/bin/env python3
'''
Write a function def np_elementwise(mat1, mat2): that performs
element-wise addition, subtraction, multiplication, and division:
You can assume that mat1 and mat2 can be interpreted as numpy.ndarrays
You should return a tuple containing the element-wise sum, difference,
product, and quotient, respectively
You are not allowed to use any loops or conditional statements
You can assume that mat1 and mat2 are never empty
'''


def np_elementwise(mat1, mat2):
    '''
    This function return addition wises
    '''
    # Element-wise addition
    sum_result = mat1 + mat2
    # Element-wise subtraction
    diff_result = mat1 - mat2
    # Element-wise multiplication
    prod_result = mat1 * mat2
    # Element-wise division
    quotient_result = mat1 / mat2
    # Return a tuple containing the results of element-wise operations
    return sum_result, diff_result, prod_result, quotient_result
