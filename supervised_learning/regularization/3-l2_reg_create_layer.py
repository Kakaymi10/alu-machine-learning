#!/usr/bin/env python3
'''
neural network reg
'''


import numpy as np
import tensorflow as tf


def l2_reg_create_layer(prev, n, activation, lambtha):
    """
    Creates a tensorflow layer that includes L2 regularization
    """
    key = "FAN_AVG"
    init = tf.contrib.layers.variance_scaling_initializer(mode=key)
    reg = tf.contrib.layers.l2_regularizer(lambtha)
    layer = tf.layers.Dense(units=n, activation=activation,
                            kernel_initializer=init,
                            kernel_regularizer=reg)
    return layer(prev)
