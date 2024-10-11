#!/usr/bin/env python3
'''
Module that contains the class SelfAttention
'''


import tensorflow as tf
import numpy as np


class SelfAttention(tf.keras.layers.Layer):
    '''
    Class that performs self-attention
    '''
    def __init__(self, units):
        '''
        Class constructor
        '''
        super(SelfAttention, self).__init__()
        self.units = units
        # Create the query, key, and value matrices
        self.W = tf.keras.layers.Dense(units)
        self.U = tf.keras.layers.Dense(units)
        self.V = tf.keras.layers.Dense(1)  # Changed to 1 unit

    def call(self, s_prev, hidden_states):
        '''
        Method that performs the self-attention
        '''
        # Expand s_prev to match hidden_states time steps
        s_prev_expanded = tf.expand_dims(s_prev, 1)
        
        # Broadcast s_prev to all time steps
        s_prev_tiled = tf.tile(s_prev_expanded, [1, tf.shape(hidden_states)[1], 1])
        
        # Compute energies
        e = self.V(tf.nn.tanh(self.W(s_prev_tiled) + self.U(hidden_states)))
        
        # Compute attention weights
        weights = tf.nn.softmax(e, axis=1)
        
        # Compute the context vector
        context = tf.reduce_sum(weights * hidden_states, axis=1)
        
        return context, weights
