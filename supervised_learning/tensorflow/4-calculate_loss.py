#!/usr/bin/env python3
""" calcualte loss of a neural network"""


import tensorflow as tf


def calculate_loss(y, y_pred):
    """ loss function with cross entropy"""
    return tf.losses.softmax_cross_entropy(y, y_pred)
