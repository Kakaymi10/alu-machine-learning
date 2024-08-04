#!/usr/bin/env python3
'''
brighten an image
'''


import tensforflow as tf
import numpy as np


def change_brightness(image, max_delta):
    '''return max delta brightened image tensor'''
    return tf.image.random_brightness(image, max_delta)