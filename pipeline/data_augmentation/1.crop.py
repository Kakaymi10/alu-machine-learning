#!/usr/bin/env python3
'''
crop image
'''


import tensforflow as tf
import numpy as np


def crop_image(image, size):
    '''return cropped image'''
    return tf.image.random_crop(image, size)