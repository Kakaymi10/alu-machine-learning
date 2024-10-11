#!/usr/bin/env python3
'''
Module that contains the class SelfAttention
'''


import tensorflow as tf


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
        # Remove concatenation of s_prev and hidden_states
        # s_prev = tf.expand_dims(s_prev, 1)
        
        # Compute energies
        e = self.V(tf.nn.tanh(self.W(s_prev) + self.U(hidden_states)))
        
        # Compute attention weights
        weights = tf.nn.softmax(e, axis=1)
        
        # Compute the context vector
        context = tf.reduce_sum(weights * hidden_states, axis=1)
        
        return context, weights
