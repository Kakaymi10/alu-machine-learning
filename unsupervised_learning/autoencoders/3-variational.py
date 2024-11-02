#!/usr/bin/env python3
'''variational autoencoder'''


import tensorflow.keras as keras


def autoencoder(input_dims, hidden_layers, latent_dims):
    '''creates an autoencoder'''
    # Encoder
    encoder_input = keras.Input(shape=(input_dims,))
    x = encoder_input
    for nodes in hidden_layers:
        x = keras.layers.Dense(nodes, activation='relu')(x)

    # Mean and log variance for latent space
    z_mean = keras.layers.Dense(latent_dims, activation=None)(x)
    z_log_var = keras.layers.Dense(latent_dims, activation=None)(x)

    # Sampling layer
    def sampling(args):
        z_mean, z_log_var = args
        batch = keras.backend.shape(z_mean)[0]
        dim = keras.backend.int_shape(z_mean)[1]
        epsilon = keras.backend.random_normal(shape=(batch, dim))
        return z_mean + keras.backend.exp(0.5 * z_log_var) * epsilon

    z = keras.layers.Lambda(sampling,
                            output_shape=(latent_dims,),
                            name='z')([z_mean, z_log_var])
    encoder = keras.models.Model(
        encoder_input, [z, z_mean, z_log_var], name='encoder'
    )

    # Decoder
    decoder_input = keras.Input(shape=(latent_dims,))
    x = decoder_input
    for nodes in reversed(hidden_layers):
        x = keras.layers.Dense(nodes, activation='relu')(x)
    x = keras.layers.Dense(input_dims, activation='sigmoid')(x)
    decoder = keras.models.Model(decoder_input, x, name='decoder')

    # Full VAE Model
    autoencoder_input = encoder_input
    z, z_mean, z_log_var = encoder(autoencoder_input)
    reconstructed = decoder(z)
    autoencoder = keras.models.Model(
        autoencoder_input, reconstructed, name='autoencoder'
    )

    # Loss function
    def vae_loss(y_true, y_pred):
        reconstruction_loss = keras.losses.binary_crossentropy(y_true, y_pred)
        reconstruction_loss *= input_dims
        a = keras.backend.square(z_mean)
        b = keras.backend.exp(z_log_var)
        kl_loss = -0.5 * keras.backend.sum(
            1 + z_log_var - a - b,
            axis=-1
        )
        return keras.backend.mean(reconstruction_loss + kl_loss)

    autoencoder.compile(optimizer='adam', loss=vae_loss)

    return encoder, decoder, autoencoder
