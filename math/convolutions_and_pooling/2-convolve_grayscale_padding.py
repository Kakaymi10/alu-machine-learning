#!/usr/bin/env python3
'''
 performs a same convolution on grayscale images:
'''


import numpy as np


def convolve_grayscale_padding(images, kernel, padding):
    '''
        A function def convolve_grayscale_padding(images, kernel, padding):
    '''
    m = images.shape[0]
    height = images.shape[1]
    width = images.shape[2]
    kh = kernel.shape[0]
    kw = kernel.shape[1]
    ph, pw = padding
    images = np.pad(images, ((0, 0), (ph, ph), (pw, pw)),
                    'constant', constant_values=0)
    ch = height + (2 * ph) - kh + 1
    cw = width + (2 * pw) - kw + 1
    convoluted = np.zeros((m, ch, cw))
    for h in range(ch):
        for w in range(cw):
            output = np.sum(images[:, h: h + kh, w: w + kw] * kernel,
                            axis=1).sum(axis=1)
            convoluted[:, h, w] = output
    return convoluted
