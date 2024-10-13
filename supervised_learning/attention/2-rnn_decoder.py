#!/usr/bin/env python3
"""
A class that inherits from tensorflow.keras.layers.Layer
to decode for machine translation
"""


import tensorflow as tf
SelfAttention = __import__('1-self_attention').SelfAttention


class RNNDecoder(tf.keras.layers.Layer):
    """
    Decodes for machine translation:
    """

    def __init__(self, vocab, embedding, units, batch):
        """
        Class constructor
        Args:
            vocab: an integer representing the size of the output vocabulary
            embedding: an integer representing the dimensionality
            of the embedding vector
            units: an integer representing the number of hidden
            units in the RNN cell
            batch: an integer representing the batch size
        Sets the following public instance attributes:
            embedding: the embedding layer for the targets
            gru: a GRU layer with units units
            F: a Dense layer with vocab units
        """
        if type(vocab) is not int:
            raise TypeError(
                "vocab must be int representing the size of output vocabulary"
            )
        if type(embedding) is not int:
            raise TypeError(
                "embedding must be int representing dimensionality of vector"
            )
        if type(units) is not int:
            raise TypeError(
                "units must be int representing the number of hidden units"
            )
        if type(batch) is not int:
            raise TypeError("batch must be int representing the batch size")
        super(RNNDecoder, self).__init__()
        self.embedding = tf.keras.layers.Embedding(
            input_dim=vocab, output_dim=embedding
        )
        self.gru = tf.keras.layers.GRU(
            units=units,
            return_state=True,
            return_sequences=True,
            recurrent_initializer="glorot_uniform",
        )
        self.F = tf.keras.layers.Dense(units=vocab)

    def call(self, x, s_prev, hidden_states):
        """
        x: a tensor of shape (batch, input_seq_len, embedding)
        containing the embedded input
        s_prev: a tensor of shape (batch, units) containing
        the previous decoder hidden state
        hidden_states: a tensor of shape (batch, input_seq_len, units)
        containing the outputs of the decoder
        Returns: y, s_next
            y: a tensor of shape (batch, target_seq_len, vocab)
            containing the generated sequences
            s_next: a tensor of shape (batch, units) containing
            the next decoder hidden state
        """
        units = s_prev.get_shape().as_list()[1]
        attention = SelfAttention(units)
        context, weights = attention(s_prev, hidden_states)
        x = self.embedding(x)
        context = tf.expand_dims(context, 1)
        x = tf.concat([context, x], axis=-1)
        y, s = self.gru(x)
        y = tf.reshape(y, (-1, y.shape[2]))
        y = self.F(y)
        return y, s
