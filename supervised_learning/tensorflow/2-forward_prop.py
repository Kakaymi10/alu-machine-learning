#!/usr/bin/env python3
"""
Neural Class
"""


import tensorflow as tf
create_layer = __import__('1-create_layer').create_layer


def forward_prop(x, layer_sizes=[], activations=[]):
    """creates the forward propagation graph for the nueral network."""
    L = x
    for i in range(len(layer_sizes)):
        L = create_layer(L, layer_sizes[i],
                         activations[i])
    return L
