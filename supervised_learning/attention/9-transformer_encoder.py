#!/usr/bin/env python3


import tensorflow as tf
positional_encoding = __import__('4-positional_encoding').positional_encoding
EncoderBlock = __import__('7-transformer_encoder_block').EncoderBlock


class Encoder(tf.keras.layers.Layer):
    '''
    encoder
    '''
    def __init__(self, N, dm, h, hidden, input_vocab, max_seq_len, drop_rate=0.1):
        super(Encoder, self).__init__()
        
        # Public attributes
        self.N = N  # Number of encoder blocks
        self.dm = dm  # Dimensionality of the model

        # Embedding layer for the inputs
        self.embedding = tf.keras.layers.Embedding(input_vocab, dm)

        # Positional encoding (precomputed for maximum sequence length)
        self.positional_encoding = positional_encoding(max_seq_len, dm)

        # Dropout layer for the positional encoding
        self.dropout = tf.keras.layers.Dropout(drop_rate)

        # List of EncoderBlocks
        self.blocks = [EncoderBlock(dm, h, hidden, drop_rate) for _ in range(N)]

    def call(self, x, training, mask):
        '''
        Get sequence length from the input tensor
        '''
        seq_len = tf.shape(x)[1]

        # Apply embedding and scale by sqrt(dm)
        x = self.embedding(x) * tf.math.sqrt(tf.cast(self.dm, tf.float32))

        # Add positional encoding (ensure proper broadcasting)
        x += self.positional_encoding[tf.newaxis, :seq_len, :]

        # Apply dropout
        x = self.dropout(x, training=training)

        # Pass through each encoder block
        for block in self.blocks:
            x = block(x, training, mask)

        # Verify shape
        tf.print("Output shape:", tf.shape(x), summarize=-1)

        return x
