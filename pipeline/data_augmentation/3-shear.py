#!/usr/bin/env python3
'''
randomly shear an image
'''


import tensforflow as tf
import numpy as np


def shear_image(image, intensity):
    '''
    return randomly sheard image
    '''
    return tf.keras.preproccessing.image.random_shear(image, intensity)