#!/usr/bin/env python3
'''
change the hue of an image
'''


import tensforflow as tf
import numpy as np


def change_hue(image, delta):
    '''hue it delta time'''
    return tf.image.adjust_hue(image, delta)