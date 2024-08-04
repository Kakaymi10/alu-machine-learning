#!/usr/bin/env python3
'''
flip image
'''


import tensforflow as tf
import numpy as np


def flip_image(image):
    '''
    returned flipped image
    '''
    return tf.image.flip_left_right(image)