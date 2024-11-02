#!/usr/bin/env python3
'''convolutional autoencoder'''


import tensorflow.keras as keras


def autoencoder(input_dims, filters, latent_dims):
    '''creates an autoencoder'''
    # Encoder
    inputs = keras.Input(shape=input_dims)
    x = inputs
    for f in filters:
        x = keras.layers.Conv2D(f, (3, 3), activation='relu', padding='same')(x)
        x = keras.layers.MaxPooling2D((2, 2), padding='same')(x)
    
    # Latent space
    latent = keras.layers.Conv2D(latent_dims[0], (3, 3), activation='relu', padding='same')(x)
    
    # Decoder
    x = latent
    for f in reversed(filters[:-1]):
        x = keras.layers.Conv2D(f, (3, 3), activation='relu', padding='same')(x)
        x = keras.layers.UpSampling2D((2, 2))(x)
    
    x = keras.layers.Conv2D(filters[0], (3, 3), activation='relu', padding='valid')(x)
    x = keras.layers.UpSampling2D((2, 2))(x)
    
    outputs = keras.layers.Conv2D(input_dims[-1], (3, 3), activation='sigmoid', padding='same')(x)
    
    # Models
    encoder = keras.Model(inputs, latent, name='encoder')
    decoder = keras.Model(latent, outputs, name='decoder')
    auto = keras.Model(inputs, decoder(encoder(inputs)), name='autoencoder')
    
    # Compile autoencoder
    auto.compile(optimizer='adam', loss='binary_crossentropy')
    
    return encoder, decoder, auto
