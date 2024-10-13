#!/usr/bin/env python3
'''
scaled attention
'''


import tensorflow as tf


def sdp_attention(Q, K, V, mask=None):
    """Calculates the scaled dot product attention."""
    # Calculate Q * K^T (using matrix multiplication)
    matmul_qk = tf.matmul(Q, K, transpose_b=True)

    # Scale by the square root of the dimension of the keys (dk)
    dk = tf.cast(tf.shape(K)[-1], tf.float32)
    scaled_qk = matmul_qk / tf.math.sqrt(dk)

    # Apply mask if provided
    if mask is not None:
        scaled_qk += (mask * -1e9)  # Masking

    # Apply softmax to get the attention weights
    weights = tf.nn.softmax(scaled_qk, axis=-1)

    # Compute the output by multiplying the weights with the value matrix V
    output = tf.matmul(weights, V)

    return output, weights
