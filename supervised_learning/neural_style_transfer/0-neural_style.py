#!/usr/bin/env python3
"""
This module implements Neural Style Transfer (NST) using TensorFlow and NumPy.
"""

import numpy as np
import tensorflow as tf


class NST:
    """
    NST class that performs Neural Style Transfer tasks.
    
    Attributes:
        style_layers (list): List of layers for style extraction.
        content_layer (str): Layer used for content extraction.
    """

    # Public class attributes
    style_layers = ['block1_conv1', 'block2_conv1', 'block3_conv1', 'block4_conv1', 'block5_conv1']
    content_layer = 'block5_conv2'

    def __init__(self, style_image, content_image, alpha=1e4, beta=1):
        """
        Initialize the NST class with style and content images, and weights for style and content cost.

        Args:
            style_image (np.ndarray): The style reference image.
            content_image (np.ndarray): The content reference image.
            alpha (float): The weight for the content cost. Default is 1e4.
            beta (float): The weight for the style cost. Default is 1.

        Raises:
            TypeError: If style_image or content_image is not a numpy array of shape (h, w, 3).
            TypeError: If alpha or beta is not a non-negative number.
        """
        if not isinstance(style_image, np.ndarray) or style_image.shape[-1] != 3:
            raise TypeError("style_image must be a numpy.ndarray with shape (h, w, 3)")
        if not isinstance(content_image, np.ndarray) or content_image.shape[-1] != 3:
            raise TypeError("content_image must be a numpy.ndarray with shape (h, w, 3)")
        if not (isinstance(alpha, (int, float)) and alpha >= 0):
            raise TypeError("alpha must be a non-negative number")
        if not (isinstance(beta, (int, float)) and beta >= 0):
            raise TypeError("beta must be a non-negative number")

        self.style_image = self.scale_image(style_image)
        self.content_image = self.scale_image(content_image)
        self.alpha = alpha
        self.beta = beta

        tf.compat.v1.enable_eager_execution()
        print("Eager execution:", tf.executing_eagerly())

    @staticmethod
    def scale_image(image):
        """
        Scales an image so that its largest side is 512 pixels, and pixel values are between 0 and 1.

        Args:
            image (np.ndarray): The image to be scaled.

        Returns:
            tf.Tensor: Scaled image of shape (1, h_new, w_new, 3).

        Raises:
            TypeError: If image is not a numpy array of shape (h, w, 3).
        """
        if not isinstance(image, np.ndarray) or image.shape[-1] != 3:
            raise TypeError("image must be a numpy.ndarray with shape (h, w, 3)")

        h, w, _ = image.shape
        if h > w:
            new_h = 512
            new_w = int(w * 512 / h)
        else:
            new_w = 512
            new_h = int(h * 512 / w)

        image_resized = tf.image.resize(image, [new_h, new_w], method=tf.image.ResizeMethod.BICUBIC)
        image_scaled = image_resized / 255.0
        image_tensor = tf.expand_dims(image_scaled, axis=0)
        
        # Print shape and min/max for debugging
        print("Scaled image shape:", image_tensor.shape)
        print("Min pixel value:", tf.reduce_min(image_tensor))
        print("Max pixel value:", tf.reduce_max(image_tensor))

        return image_tensor
