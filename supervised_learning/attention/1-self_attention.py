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
        self.W = tf.keras.layers.Dense(units)
        self.U = tf.keras.layers.Dense(units)
        self.V = tf.keras.layers.Dense(1)

    def call(self, s_prev, hidden_states):
        '''
        Method that performs the self-attention
        '''
        # Expand s_prev to match hidden_states time steps
        s_prev_expanded = tf.expand_dims(s_prev, 1)

        # Compute energies
        e = self.V(tf.nn.tanh(self.W(s_prev_expanded) + self.U(hidden_states)))

        # Compute attention weights
        weights = tf.nn.softmax(e, axis=1)

        # Compute the context vector
        context = tf.reduce_sum(weights * hidden_states, axis=1)

        return context, weights
