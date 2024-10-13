#!/usr/bin/env python3
'''
Transformer encoder block
'''


import tensorflow as tf
MultiHeadAttention = __import__('6-multihead_attention').MultiHeadAttention


class EncoderBlock(tf.keras.layers.Layer):
    '''
    encoder block
    '''
    def __init__(self, dm, h, hidden, drop_rate=0.1):
        """Initialize the EncoderBlock."""
        super(EncoderBlock, self).__init__()
        self.mha = MultiHeadAttention(dm, h)  # Multi-Head Attention layer

        # Fully connected layers
        self.dense_hidden = tf.keras.layers.Dense(hidden, activation='relu')
        self.dense_output = tf.keras.layers.Dense(dm)  # Output dense layer

        # Layer Normalization with epsilon for numerical stability
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        # Dropout layers for regularization
        self.dropout1 = tf.keras.layers.Dropout(drop_rate)
        self.dropout2 = tf.keras.layers.Dropout(drop_rate)

    def call(self, x, training, mask=None):
        """
        Execute the encoder block logic.

        Args:
            x: Tensor of shape (batch, input_seq_len, dm).
            training: Boolean, whether the model is in training mode.
            mask: Mask to apply in the multi-head attention.
        Returns:
            Tensor of shape (batch, input_seq_len, dm).
        """
        # 1. Multi-head attention with residual connection and layer norm
        attn_output, _ = self.mha(x, x, x, mask)  # Self-attention
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(x + attn_output)

        # 2. Feed-forward network with residual connection and layer norm
        hidden_output = self.dense_hidden(out1)  # Hidden dense layer
        ffn_output = self.dense_output(hidden_output)  # Output dense layer
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.layernorm2(out1 + ffn_output)

        return out2  # Return the final output
