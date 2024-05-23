#!/usr/bin/env python3
"""
Neural Class
"""


import tensorflow as tf
create_layer = __import__('1-create_layer').create_layer


def forward_prop(x, layer_sizes=[], activations=[]):
    '''
    creates the forward propagation graph for the nueral network.
    '''
    # input data
    output = x

    # loop thorugh the layers and create each one
    for i in range(len(layer_sizes)):
        outout = create_layer(output, layer_sizes[i], activations[i])

    return output
