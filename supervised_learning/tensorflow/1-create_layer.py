#!/usr/bin/env python3
"""
Neural Class
"""


import tensorflow as tf


def create_layer(prev, n, activation):
    """
    Creates a layer for a neural network
    """
    # Initialize the weights using He et al. initialization
    W = tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG")
    L = tf.layers.Dense(units=n, activation=activation,
                        kernel_initializer=W, name="layer")
    return L(prev)
