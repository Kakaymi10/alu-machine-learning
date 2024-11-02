#!/usr/bin/env python3
'''sparse autoencoder'''


import tensorflow.keras as keras


def autoencoder(input_dims, hidden_layers, latent_dims, lambtha):
    '''creates an autoencoder'''
    # Encoder
    input_layer = keras.Input(shape=(input_dims,))
    encoded = input_layer
    for nodes in hidden_layers:
        encoded = keras.layers.Dense(nodes, activation='relu')(encoded)
    encoded = keras.layers.Dense(
        latent_dims,
        activation='relu',
        activity_regularizer=keras.regularizers.l1(lambtha)
    )(encoded)

    # Decoder
    decoded = encoded
    for nodes in reversed(hidden_layers):
        decoded = keras.layers.Dense(nodes, activation='relu')(decoded)
    decoded = keras.layers.Dense(input_dims, activation='sigmoid')(decoded)

    # Models
    encoder = keras.Model(input_layer, encoded)
    decoder_input = keras.Input(shape=(latent_dims,))
    decoder_layer = decoder_input
    for nodes in reversed(hidden_layers):
        decoder_layer = keras.layers.Dense(
            nodes, activation='relu'
        )(decoder_layer)
    decoder_layer = keras.layers.Dense(
        input_dims, activation='sigmoid'
    )(decoder_layer)
    decoder = keras.Model(decoder_input, decoder_layer)

    auto_input = keras.Input(shape=(input_dims,))
    encoded_auto = encoder(auto_input)
    decoded_auto = decoder(encoded_auto)
    auto = keras.Model(auto_input, decoded_auto)

    auto.compile(optimizer='adam', loss='binary_crossentropy')

    return encoder, decoder, auto
