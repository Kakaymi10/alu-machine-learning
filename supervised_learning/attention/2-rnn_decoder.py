#!/usr/bin/env python3
"""RNNDecoder module for machine translation."""


import tensorflow as tf
SelfAttention = __import__('1-self_attention').SelfAttention


class RNNDecoder(tf.keras.layers.Layer):
    """RNNDecoder class to decode sequences for machine translation."""
    def __init__(self, vocab, embedding, units, batch):
        """
        Initializes the RNNDecoder layer.

        Args:
            vocab (int): Size of the output vocabulary.
            embedding (int): Dimensionality of the embedding vector.
            units (int): Number of hidden units in the RNN cell.
            batch (int): Batch size.
        """
        super(RNNDecoder, self).__init__()

        self.embedding = tf.keras.layers.Embedding(
            input_dim=vocab, output_dim=embedding
        )
        self.gru = tf.keras.layers.GRU(
            units, return_sequences=True, return_state=True,
            recurrent_initializer='glorot_uniform'
        )
        self.F = tf.keras.layers.Dense(vocab)
        self.attention = SelfAttention(units)

    def call(self, x, s_prev, hidden_states):
        """
        Performs the forward pass for the decoder.

        Args:
            x (tf.Tensor):
            Tensor of shape (batch, 1) representing the previous
            word in the target sequence as an index.
            s_prev (tf.Tensor):
            Tensor of shape (batch, units) representing the
            previous decoder hidden state.
            hidden_states (tf.Tensor):
            Tensor of shape (batch, input_seq_len, units)
            representing the encoder outputs.
        Returns:
            y (tf.Tensor):
            Tensor of shape (batch, vocab) with output word as
            a one-hot vector in the target vocabulary.
            s (tf.Tensor):
            Tensor of shape (batch, units) with the new hidden state.
        """
        # Embed the input word
        x = self.embedding(x)

        # Compute the context vector using self-attention
        context_vector, _ = self.attention(s_prev, hidden_states)

        # Concatenate context vector with the embedded input
        context_vector = tf.expand_dims(context_vector, 1)
        x = tf.concat([context_vector, x], axis=-1)

        # Pass the concatenated input through the GRU layer
        output, s = self.gru(x, initial_state=s_prev)

        # Generate output probabilities with the Dense layer
        y = self.F(output)

        # Remove the extra sequence dimension from y
        y = tf.squeeze(y, axis=1)

        return y, s
