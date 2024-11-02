#!/usr/bin/env python3
'''convolutional autoencoder'''


from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D
from tensorflow.keras import Model


def autoencoder(input_dims, filters, latent_dims):
    '''creates an autoencoder'''
    # Encoder
    inputs = Input(shape=input_dims)
    x = inputs
    for f in filters:
        x = Conv2D(f, (3, 3), activation='relu', padding='same')(x)
        x = MaxPooling2D((2, 2), padding='same')(x)
    
    # Latent space
    latent = Conv2D(latent_dims[0], (3, 3), activation='relu', padding='same')(x)
    
    # Decoder
    x = latent
    for f in reversed(filters[:-1]):
        x = Conv2D(f, (3, 3), activation='relu', padding='same')(x)
        x = UpSampling2D((2, 2))(x)
    
    x = Conv2D(filters[0], (3, 3), activation='relu', padding='valid')(x)
    x = UpSampling2D((2, 2))(x)
    
    outputs = Conv2D(input_dims[-1], (3, 3), activation='sigmoid', padding='same')(x)
    
    # Models
    encoder = Model(inputs, latent, name='encoder')
    decoder = Model(latent, outputs, name='decoder')
    auto = Model(inputs, decoder(encoder(inputs)), name='autoencoder')
    
    # Compile autoencoder
    auto.compile(optimizer='adam', loss='binary_crossentropy')
    
    return encoder, decoder, auto
