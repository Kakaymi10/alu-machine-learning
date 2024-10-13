#!/usr/bin/env python3
"""
Defines a class that inherits from tensorflow.keras.layers.Layer
to create the encoder for a transformer
"""


import tensorflow as tf
positional_encoding = __import__('4-positional_encoding').positional_encoding
EncoderBlock = __import__('7-transformer_encoder_block').EncoderBlock


class Encoder(tf.keras.layers.Layer):
    """
    Class to create the encoder for a transformer
    """
    def __init__(self, N, dm, h, hidden, input_vocab, max_seq_len,
                 drop_rate=0.1):
        """
        Class constructor
        """
        if type(N) is not int:
            raise TypeError(
                "N must be int representing number of blocks in the encoder")
        if type(dm) is not int:
            raise TypeError(
                "dm must be int representing dimensionality of model")
        if type(h) is not int:
            raise TypeError(
                "h must be int representing number of heads")
        if type(hidden) is not int:
            raise TypeError(
                "hidden must be int representing number of hidden units")
        if type(input_vocab) is not int:
            raise TypeError(
                "input_vocab must be int representing size of input vocab")
        if type(max_seq_len) is not int:
            raise TypeError(
                "max_seq_len must be int representing max sequence length")
        if type(drop_rate) is not float:
            raise TypeError(
                "drop_rate must be float representing dropout rate")
        super(Encoder, self).__init__()
        self.N = N
        self.dm = dm
        self.embedding = tf.keras.layers.Embedding(input_dim=input_vocab,
                                                   output_dim=dm)
        self.positional_encoding = positional_encoding(max_seq_len, dm)
        self.blocks = [EncoderBlock(dm, h, hidden, drop_rate)
                       for block in range(N)]
        self.dropout = tf.keras.layers.Dropout(drop_rate)

    def call(self, x, training, mask):
        """
        Calls the encoder and returns the encoder's output

        parameters:
            x [tensor of shape (batch, input_seq_len, dm)]:
                contains the input to the encoder
            training [boolean]:
                determines if the model is in training
            mask:
                mask to be applied for multi-head attention
        """
        seq_len = x.shape[1]

        x = self.embedding(x)
        x *= tf.math.sqrt(tf.cast(self.dm, tf.float32))
        x += self.positional_encoding[:seq_len, :]
        x = self.dropout(x, training=training)

        for i in range(self.N):
            x = self.blocks[i](x, training, mask)

        return x
