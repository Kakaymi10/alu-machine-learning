#!/usr/bin/env python3
'''convolutional autoencoder'''


import tensorflow.keras as keras


def autoencoder(input_dims, filters, latent_dims):
    # Encoder
    encoder_input = keras.Input(shape=input_dims)
    x = encoder_input
    for filter_size in filters:
        x = keras.layers.Conv2D(filter_size, (3, 3), padding='same', activation='relu')(x)
        x = keras.layers.MaxPooling2D((2, 2), padding='same')(x)
    x = keras.layers.Conv2D(latent_dims[0], (3, 3), padding='same', activation='relu')(x)
    encoder_output = x
    encoder = keras.models.Model(encoder_input, encoder_output, name='encoder')
    
    # Decoder
    decoder_input = keras.Input(shape=latent_dims)
    x = decoder_input
    for filter_size in reversed(filters[:-1]):
        x = keras.layers.Conv2D(filter_size, (3, 3), padding='same', activation='relu')(x)
        x = keras.layers.UpSampling2D((2, 2))(x)
    x = keras.layers.Conv2D(filters[0], (3, 3), padding='valid', activation='relu')(x)
    x = keras.layers.Conv2D(input_dims[-1], (3, 3), activation='sigmoid', padding='same')(x)
    decoder_output = x
    decoder = keras.models.Model(decoder_input, decoder_output, name='decoder')
    
    # Autoencoder
    autoencoder_input = encoder_input
    encoded = encoder(autoencoder_input)
    decoded = decoder(encoded)
    autoencoder = keras.models.Model(autoencoder_input, decoded, name='autoencoder')
    
    # Compile the model
    autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
    
    return encoder, decoder, autoencoder
