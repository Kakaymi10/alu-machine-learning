#1/usr/bin/env python3
'''
regularize me
'''


import numpy as np
import tensorflow as tf

def l2_reg_cost(cost):
    """
    Calculates the cost of a neural network with L2 regularization
    """
    # Get the regularization losses added to the graph
    regularization_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
    
    # Add the regularization losses to the original cost
    l2_cost = cost + tf.reduce_sum(regularization_losses)
    
    return l2_cost
