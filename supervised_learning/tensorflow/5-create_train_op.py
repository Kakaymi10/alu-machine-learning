#!/usr/bin/env python3
""" train a neural network"""


import tensorflow as tf


def create_train_op(loss, alpha):
    """ training operation of a NN with Grad descent"""
    return tf.train.GradientDescentOptimizer(alpha).minimize(loss)
