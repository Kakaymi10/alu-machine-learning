#!/usr/bin/env python3
'''
Module that contains the class RNNDecoder
'''


import tensorflow as tf

SelfAttention = __import__('1-self_attention').SelfAttention

class RNNDecoder(tf.keras.layers.Layer):
    def __init__(self, vocab, embedding, units, batch):
        super(RNNDecoder, self).__init__()
        self.embedding = tf.keras.layers.Embedding(vocab, embedding,
                                                   embeddings_initializer='glorot_uniform')
        self.gru = tf.keras.layers.GRU(units,
                                       return_sequences=True,
                                       return_state=True,
                                       recurrent_initializer='glorot_uniform',
                                       kernel_initializer='glorot_uniform',
                                       bias_initializer='zeros')
        self.F = tf.keras.layers.Dense(vocab,
                                       kernel_initializer='glorot_uniform',
                                       bias_initializer='zeros')
        self.attention = SelfAttention(units)

    def call(self, x, s_prev, hidden_states):
        # x shape: (batch, 1)
        # s_prev shape: (batch, units)
        # hidden_states shape: (batch, input_seq_len, units)
        
        x = self.embedding(x)  # (batch, 1, embedding)
        context, _ = self.attention(s_prev, hidden_states)
        x = tf.concat([tf.expand_dims(context, 1), x], axis=-1)
        
        output, state = self.gru(x, initial_state=s_prev)
        
        output = tf.reshape(output, (-1, output.shape[2]))
        y = self.F(output)
        
        return y, state


