#!/usr/bin/env python3
'''
Function def create_RMSProp_op(loss, alpha, beta2, epsilon)
that generates the training operation for a neural network in TensorFlow
using the RMSProp optimization algorithm:
'''


import tensorflow as tf


def create_RMSProp_op(loss, alpha, beta2, epsilon):
    '''
    Generates the training operation for a neural network in TensorFlow
    using the RMSProp optimization algorithm
    '''
    optimizer = tf.train.RMSPropOptimizer(alpha, beta2, epsilon=epsilon)
    train_op = optimizer.minimize(loss)
    return train_op
