#!/usr/bin/env python3
'''sparse autoencoder'''


from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l1
import tensorflow.keras as keras


def autoencoder(input_dims, hidden_layers, latent_dims, lambtha):
    # Encoder
    input_layer = Input(shape=(input_dims,))
    encoded = input_layer
    for nodes in hidden_layers:
        encoded = Dense(nodes, activation='relu')(encoded)
    encoded = Dense(latent_dims, activation='relu', activity_regularizer=l1(lambtha))(encoded)
    
    # Decoder
    decoded = encoded
    for nodes in reversed(hidden_layers):
        decoded = Dense(nodes, activation='relu')(decoded)
    decoded = Dense(input_dims, activation='sigmoid')(decoded)
    
    # Models
    encoder = Model(input_layer, encoded)
    decoder_input = Input(shape=(latent_dims,))
    decoder_layer = decoder_input
    for nodes in reversed(hidden_layers):
        decoder_layer = Dense(nodes, activation='relu')(decoder_layer)
    decoder_layer = Dense(input_dims, activation='sigmoid')(decoder_layer)
    decoder = Model(decoder_input, decoder_layer)
    
    auto_input = Input(shape=(input_dims,))
    encoded_auto = encoder(auto_input)
    decoded_auto = decoder(encoded_auto)
    auto = Model(auto_input, decoded_auto)
    
    auto.compile(optimizer='adam', loss='binary_crossentropy')
    
    return encoder, decoder, auto