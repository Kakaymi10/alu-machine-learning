#!/usr/bin/env python3
'''
multi headed attention
'''


import tensorflow as tf
sdp_attention = __import__('5-sdp_attention').sdp_attention


class MultiHeadAttention(tf.keras.layers.Layer):
    '''
    Multi headed att
    '''
    def __init__(self, dm, h):
        """Class constructor for multi-head attention."""
        super(MultiHeadAttention, self).__init__()
        assert dm % h == 0, "dm must be divisible by h"

        self.dm = dm  # Dimensionality of the model
        self.h = h    # Number of heads
        self.depth = dm // h  # Depth of each attention head

        # Define Dense layers for Q, K, V and the final linear layer
        self.Wq = tf.keras.layers.Dense(dm)
        self.Wk = tf.keras.layers.Dense(dm)
        self.Wv = tf.keras.layers.Dense(dm)
        self.linear = tf.keras.layers.Dense(dm)

    def split_heads(self, x, batch_size):
        """Split the last dimension into (h, depth) and transpose
        for multi-head attention."""
        x = tf.reshape(x, (batch_size, -1, self.h, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, Q, K, V, mask):
        """Executes the multi-head attention mechanism."""
        batch_size = tf.shape(Q)[0]  # Get the batch size

        # Generate Q, K, V matrices using the Dense layers
        Q = self.Wq(Q)  # Shape: (batch, seq_len_q, dm)
        K = self.Wk(K)  # Shape: (batch, seq_len_v, dm)
        V = self.Wv(V)  # Shape: (batch, seq_len_v, dm)

        # Split Q, K, V into multiple heads
        Q = self.split_heads(Q, batch_size)
        K = self.split_heads(K, batch_size)
        V = self.split_heads(V, batch_size)

        # Compute scaled dot product attention for each head
        scaled_attention, weights = sdp_attention(Q, K, V, mask)

        # Transpose and reshape the output back to (batch, seq_len_q, dm)
        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])
        concat_att = tf.reshape(scaled_attention, (batch_size, -1, self.dm))
        # Apply the final linear layer to get the attention output
        output = self.linear(concat_att)  # (batch, seq_len_q, dm)

        return output, weights
