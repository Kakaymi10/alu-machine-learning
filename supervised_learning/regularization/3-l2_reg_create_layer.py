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
    # Define the L2 regularizer
    l2_regularizer = tf.contrib.layers.l2_regularizer(lambtha)
    
    # Create a Dense layer with L2 regularization
    layer = tf.layers.Dense(
        units=n,
        activation=activation,
        kernel_initializer=tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG"),
        kernel_regularizer=l2_regularizer
    )
    
    # Apply the layer to the input
    return layer(prev)
