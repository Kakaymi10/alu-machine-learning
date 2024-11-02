#!/usr/bin/env python3
"""Vanilla GAN"""


import tensorflow as tf


def autoencoder(input_dims, hidden_layers, latent_dims):
    '''creates an autoencoder'''
    # Encoder
    encoder_input = tf.keras.Input(shape=(input_dims,))
    x = encoder_input
    for nodes in hidden_layers:
        x = tf.keras.layers.Dense(nodes, activation='relu')(x)
    latent = tf.keras.layers.Dense(latent_dims, activation='relu')(x)
    encoder = tf.keras.Model(encoder_input, latent)

    # Decoder
    decoder_input = tf.keras.Input(shape=(latent_dims,))
    x = decoder_input
    for nodes in reversed(hidden_layers):
        x = tf.keras.layers.Dense(nodes, activation='relu')(x)
    decoder_output = tf.keras.layers.Dense(input_dims, activation='sigmoid')(x)
    decoder = tf.keras.Model(decoder_input, decoder_output)

    # Autoencoder
    autoencoder_input = tf.keras.Input(shape=(input_dims,))
    encoded = encoder(autoencoder_input)
    decoded = decoder(encoded)
    auto = tf.keras.Model(autoencoder_input, decoded)

    auto.compile(optimizer='adam', loss='binary_crossentropy')

    return encoder, decoder, auto