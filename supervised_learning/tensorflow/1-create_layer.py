#!/usr/bin/env python3
"""
Neural Class
"""


import numpy as np
import tensorflow as tf


def create_layer(prev, n, activation):
    """
    Creates a layer for a neural network
    """
    # Initialize the weights using He et al. initialization
    initializer = tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG")

    # Create the dense layer with the specified number of nodes an actfunc
    layer = tf.layers.dense(inputs=prev, units=n, activation=activation,
                            kernel_initializer=initializer, name='layer')
    
    return layer

