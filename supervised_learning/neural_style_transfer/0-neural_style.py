#!/usr/bin/env python3
"""
Defines the NST class that performs Neural Style Transfer.
"""


import numpy as np
import tensorflow as tf


class NST:
    # Public class attributes for style and content layers
    style_layers = [
        'block1_conv1',
        'block2_conv1',
        'block3_conv1',
        'block4_conv1',
        'block5_conv1'
    ]
    content_layer = 'block5_conv2'

    def __init__(self, style_image, content_image, alpha=1e4, beta=1):
        """
        Initializes the NST object with the given style image, content image, and parameters.

        Args:
        style_image (np.ndarray): The style reference image.
        content_image (np.ndarray): The content reference image.
        alpha (float): Weight for content cost.
        beta (float): Weight for style cost.
        """
        # Check if the style_image is a numpy array of shape (h, w, 3)
        if not isinstance(style_image, np.ndarray) or style_image.shape[-1] != 3:
            raise TypeError("style_image must be a numpy.ndarray with shape (h, w, 3)")

        # Check if the content_image is a numpy array of shape (h, w, 3)
        if not isinstance(content_image, np.ndarray) or content_image.shape[-1] != 3:
            raise TypeError("content_image must be a numpy.ndarray with shape (h, w, 3)")

        # Check if alpha and beta are non-negative numbers
        if not (isinstance(alpha, (int, float)) and alpha >= 0):
            raise ValueError("alpha must be a non-negative number")
        if not (isinstance(beta, (int, float)) and beta >= 0):
            raise ValueError("beta must be a non-negative number")

        # Enable eager execution for TensorFlow
        tf.config.run_functions_eagerly(True)

        # Set instance attributes
        self.style_image = NST.scale_image(style_image)  # Preprocessed style image
        self.content_image = NST.scale_image(content_image)  # Preprocessed content image
        self.alpha = alpha  # Weight for content cost
        self.beta = beta  # Weight for style cost

    @staticmethod
    def scale_image(image):
        """
        Scales an image such that its pixel values are between 0 and 1
        and its largest side is 512 pixels.

        Args:
        image (np.ndarray): The image to be scaled.

        Returns:
        tf.Tensor: The scaled image.
        """
        # Check if the input is a valid image of shape (h, w, 3)
        if not isinstance(image, np.ndarray) or image.shape[-1] != 3:
            raise TypeError("image must be a numpy.ndarray with shape (h, w, 3)")

        # Get original dimensions of the image
        h, w, _ = image.shape

        # Compute scaling factors for resizing
        max_dim = 512
        scale = max_dim / max(h, w)
        new_h = int(h * scale)
        new_w = int(w * scale)

        # Resize the image using bicubic interpolation
        image = tf.image.resize(image, (new_h, new_w), method='bicubic')

        # Rescale pixel values to [0, 1]
        image = image / 255.0

        # Add a batch dimension and return
        scaled_image = tf.expand_dims(image, axis=0)
        return scaled_image
