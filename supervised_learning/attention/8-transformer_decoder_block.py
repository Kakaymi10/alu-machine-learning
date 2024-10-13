#!/usr/bin/env python3
'''
Transfromer decoder block
'''


import tensorflow as tf
MultiHeadAttention = __import__('6-multihead_attention').MultiHeadAttention


class DecoderBlock(tf.keras.layers.Layer):
    '''
    decoder block
    '''
    def __init__(self, dm, h, hidden, drop_rate=0.1):
        super(DecoderBlock, self).__init__()

        # Initialize the two multi-head attention layers
        self.mha1 = MultiHeadAttention(dm, h)  # First MHA
        self.mha2 = MultiHeadAttention(dm, h)  # Second MHA (cross-attention)

        # Dense layers: hidden (with relu) and output layers
        self.dense_hidden = tf.keras.layers.Dense(hidden, activation='relu')
        self.dense_output = tf.keras.layers.Dense(dm)

        # Layer Normalization layers
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        # Dropout layers
        self.dropout1 = tf.keras.layers.Dropout(drop_rate)
        self.dropout2 = tf.keras.layers.Dropout(drop_rate)
        self.dropout3 = tf.keras.layers.Dropout(drop_rate)

    def call(self, x, encoder_output, training, look_ahead_mask, padding_mask):
        '''
        1st Multi-Head Attention (Self-attention) + Dropout + Layer Norm
        '''
        attn1, _ = self.mha1(x, x, x, mask=look_ahead_mask)
        attn1 = self.dropout1(attn1, training=training)
        out1 = self.layernorm1(x + attn1)  # Residual connection

        # 2nd Multi-Head Attention (Cross-attention) + Dropout + Layer Norm
        attn2, _ = self.mha2(out1,
                             encoder_output,
                             encoder_output,
                             mask=padding_mask)
        attn2 = self.dropout2(attn2, training=training)
        out2 = self.layernorm2(out1 + attn2)  # Residual connection

        # Feedforward Dense layers + Dropout + Layer Norm
        hidden_output = self.dense_hidden(out2)
        dense_output = self.dense_output(hidden_output)
        dense_output = self.dropout3(dense_output, training=training)
        out3 = self.layernorm3(out2 + dense_output)  # Residual connection

        return out3
