#!/usr/bin/env python3
"""
Performs a valid convolution on grayscale images images
"""

import numpy as np


def convolve_grayscale_valid(images, kernel):
    """ convolve grayscale images

    Args:
        images (_type_): _description_
        kernel (_type_): _description_
    """

    m, h, w = images.shape
    kh, kw = kernel.shape
    output_h = h - kh + 1
    output_w = w - kw + 1
    output = np.zeros((m, output_h, output_w))
    for i in range(output_h):
        for j in range(output_w):
            output[:, i, j] = (kernel * images[:, i: i + kh, j: j + kw])\
                .sum(axis=(1, 2))
    return output
