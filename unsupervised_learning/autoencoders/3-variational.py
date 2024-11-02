#!/usr/bin/env python3
'''variational autoencoder'''


import tensorflow.keras as keras


def sampling(args):
    z_mean, z_log_var = args
    batch = keras.backend.shape(z_mean)[0]
    dim = keras.backend.int_shape(z_mean)[1]
    epsilon = keras.backend.random_normal(shape=(batch, dim))
    return z_mean + keras.backend.exp(0.5 * z_log_var) * epsilon

def autoencoder(input_dims, hidden_layers, latent_dims):
    # Encoder
    inputs = keras.layers.Input(shape=(input_dims,))
    x = inputs
    for nodes in hidden_layers:
        x = keras.layers.Dense(nodes, activation='relu')(x)
    z_mean = keras.layers.Dense(latent_dims, activation=None)(x)
    z_log_var = keras.layers.Dense(latent_dims, activation=None)(x)
    z = keras.layers.Lambda(sampling, output_shape=(latent_dims,))([z_mean, z_log_var])
    encoder = keras.models.Model(inputs, [z, z_mean, z_log_var], name='encoder')

    # Decoder
    latent_inputs = keras.layers.Input(shape=(latent_dims,))
    x = latent_inputs
    for nodes in reversed(hidden_layers):
        x = keras.layers.Dense(nodes, activation='relu')(x)
    outputs = keras.layers.Dense(input_dims, activation='sigmoid')(x)
    decoder = keras.models.Model(latent_inputs, outputs, name='decoder')

    # Autoencoder
    autoencoder_outputs = decoder(encoder(inputs)[0])
    auto = keras.models.Model(inputs, autoencoder_outputs, name='autoencoder')

    # Loss function
    reconstruction_loss = keras.losses.binary_crossentropy(inputs, autoencoder_outputs)
    reconstruction_loss *= input_dims
    kl_loss = 1 + z_log_var - keras.backend.square(z_mean) - keras.backend.exp(z_log_var)
    kl_loss = keras.backend.sum(kl_loss, axis=-1)
    kl_loss *= -0.5
    vae_loss = keras.backend.mean(reconstruction_loss + kl_loss)
    auto.add_loss(vae_loss)

    auto.compile(optimizer='adam')

    return encoder, decoder, auto
