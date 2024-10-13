#!/usr/bin/env python3


import tensorflow as tf
SelfAttention = __import__('1-self_attention').SelfAttention


class RNNDecoder(tf.keras.layers.Layer):
    def __init__(self, vocab, embedding, units, batch):
        super(RNNDecoder, self).__init__()

        # Public instance attributes
        self.embedding = tf.keras.layers.Embedding(input_dim=vocab, output_dim=embedding)
        self.gru = tf.keras.layers.GRU(
            units,
            return_sequences=True,
            return_state=True,
            recurrent_initializer='glorot_uniform'
        )
        self.F = tf.keras.layers.Dense(vocab)
        self.units = units

        # Initialize SelfAttention
        self.attention = SelfAttention(units)

    def call(self, x, s_prev, hidden_states):
        # 1. Embed the input word
        x = self.embedding(x)

        # 2. Use Self-Attention to compute context vector
        context_vector, _ = self.attention(s_prev, hidden_states)

        # 3. Concatenate context vector with the embedded input
        x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1)

        # 4. Pass the concatenated input through the GRU layer
        output, s = self.gru(x, initial_state=s_prev)

        # 5. Use the Dense layer to generate output probabilities
        y = self.F(output)

        # Remove the sequence dimension for y
        y = tf.squeeze(y, axis=1)

        return y, s
