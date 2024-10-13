#!/usr/bin/env python3
'''
Position encoding
'''


import numpy as np


def positional_encoding(max_seq_len, dm):
    """Calculates the positional encoding for a transformer."""
    # Create a matrix to hold the positional encoding vectors
    PE = np.zeros((max_seq_len, dm))

    # Calculate the position indices and dimension indices
    position = np.arange(max_seq_len)[:, np.newaxis]  # Shape: (max_seq_len, 1)
    div_term = np.exp(np.arange(0, dm, 2) * -(np.log(10000.0) / dm))

    # Apply sine to even indices (2i) and cosine to odd indices (2i + 1)
    PE[:, 0::2] = np.sin(position * div_term)  # Even dimensions
    PE[:, 1::2] = np.cos(position * div_term)  # Odd dimensions

    return PE
