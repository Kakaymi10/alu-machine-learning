#!/usr/bin/env python3
"""
A function that a valid convolution on grayscale
if necessary, the image should be padded with 0’s
"""

import numpy as np


def convolve_grayscale_valid(images, kernel):
    """
    convolves images on grayscal
    """

    m, h, w = images.shape
    kh, kw = kernel.shape
    
    # Calculate output dimensions
    out_h = h - kh + 1
    out_w = w - kw + 1
    
    # Initialize an empty array to store the convolved images
    convolved_images = np.zeros((m, out_h, out_w))
    
    # Perform convolution
    for i in range(m):  # Loop over each image
        for j in range(out_h):  # Loop over rows of output image
            for k in range(out_w):  # Loop over columns of output image
                # Extract the region of interest from the image
                image_patch = images[i, j:j+kh, k:k+kw]
                # Apply the convolution operation between the kernel and the image patch
                convolved_images[i, j, k] = np.sum(image_patch * kernel)
    
    return convolved_images
