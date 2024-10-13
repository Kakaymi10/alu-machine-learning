#!/usr/bin/env python3

import tensorflow as tf
positional_encoding = __import__('4-positional_encoding').positional_encoding
EncoderBlock = __import__('7-transformer_encoder_block').EncoderBlock

class Encoder(tf.keras.layers.Layer):
    """
    Encoder class for Transformer.
    """

    def __init__(self, N, dm, h, hidden, input_vocab, max_seq_len, drop_rate=0.1):
        """
        Class constructor.
        """
        super(Encoder, self).__init__()

        # Store instance attributes
        self.N = N
        self.dm = dm

        # Embedding layer
        self.embedding = tf.keras.layers.Embedding(input_dim=input_vocab, output_dim=dm)

        # Positional encoding (precomputed as a numpy array)
        self.positional_encoding = positional_encoding(max_seq_len, dm)

        # Dropout layer to apply to embeddings
        self.dropout = tf.keras.layers.Dropout(rate=drop_rate)

        # Create N encoder blocks
        self.blocks = [EncoderBlock(dm, h, hidden, drop_rate) for _ in range(N)]

    def call(self, x, training, mask):
        """
        Forward pass through the Encoder.
        """
        # Get input sequence length
        seq_len = tf.shape(x)[1]

        # Apply embedding and scale by sqrt(dm)
        x = self.embedding(x) * tf.math.sqrt(tf.cast(self.dm, tf.float32))

        # Add positional encoding (with broadcasting for batch dimension)
        x += self.positional_encoding[:seq_len]

        # Apply dropout to the input
        x = self.dropout(x, training=training)

        # Pass through each encoder block
        for block in self.blocks:
            x = block(x, training=training, mask=mask)

        return x
