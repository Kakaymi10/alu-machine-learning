#!/usr/bin/env python3
'''
rotate an image
'''


import tensforflow as tf
import numpy as np


def rotate_image(image):
    '''return rotated image
    by 90 degrees counter-clockwise'''
    return tf.image.rot90(image)