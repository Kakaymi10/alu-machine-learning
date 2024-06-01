#!/usr/bin/env python3
""" Create a Layer with Dropout """


import tensorflow as tf


def dropout_create_layer(prev, n, activation, keep_prob):
    """
    Create a neural network layer with dropout regularization.
    """
    init = tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG")
    layer = tf.layers.Dense(units=n,
                            activation=activation,
                            kernel_initializer=init)
    dropout = tf.layers.Dropout(rate=1 - keep_prob)
    return dropout(layer(prev))
