#!/usr/bin/env python3
'''Create a class NST that performs tasks for neural style transfer'''

import numpy as np
import tensorflow as tf


class NST:
    '''This class performs neural style transfer'''

    # Public class attributes
    style_layers = ['block1_conv1',
                    'block2_conv1',
                    'block3_conv1',
                    'block4_conv1',
                    'block5_conv1']

    content_layer = 'block5_conv2'

    def __init__(self, style_image, content_image, alpha=1e4, beta=1):
        '''
        Creates an instance of the NST class
        Args:
            style_image: image np.array used as style reference
            content_image: image np.array used as content reference
            alpha: the weight of the content cost
            beta: the weight of the style cost
        '''
        # Validation for input images and weights
        if not isinstance(style_image, np.ndarray) or len(style_image.shape) != 3 or style_image.shape[2] != 3:
            raise TypeError('style_image must be a numpy.ndarray with shape (h, w, 3)')
        if not isinstance(content_image, np.ndarray) or len(content_image.shape) != 3 or content_image.shape[2] != 3:
            raise TypeError('content_image must be a numpy.ndarray with shape (h, w, 3)')
        if not isinstance(alpha, (int, float)) or alpha < 0:
            raise TypeError('alpha must be a non-negative number')
        if not isinstance(beta, (int, float)) or beta < 0:
            raise TypeError('beta must be a non-negative number')

        # Enable eager execution (for older versions of TensorFlow)
        tf.enable_eager_execution()

        # Instance attributes
        self.style_image = self.scale_image(style_image)
        self.content_image = self.scale_image(content_image)
        self.alpha = alpha
        self.beta = beta

        # Load the model and generate features
        self.load_model()
        self.generate_features()

    @staticmethod
    def scale_image(image):
        '''Rescales an image such that values are between 0 and 1 and largest side is 512 pixels'''
        if not isinstance(image, np.ndarray) or len(image.shape) != 3 or image.shape[2] != 3:
            raise TypeError('image must be a numpy.ndarray with shape (h, w, 3)')

        h, w, _ = image.shape
        if h > w:
            new_h, new_w = 512, int(w * (512 / h))
        else:
            new_w, new_h = 512, int(h * (512 / w))

        image = tf.image.resize_bicubic(np.expand_dims(image, axis=0), size=(new_h, new_w))
        image = image / 255.0
        image = tf.clip_by_value(image, 0.0, 1.0)
        return image

    def load_model(self):
        '''Creates the model used to calculate the style and content loss'''
        # Load pre-trained VGG19 model with ImageNet weights
        vgg = tf.keras.applications.VGG19(include_top=False, weights='imagenet')
        vgg.trainable = False

        # Extract outputs of the style and content layers
        style_outputs = [vgg.get_layer(name).output for name in self.style_layers]
        content_output = vgg.get_layer(self.content_layer).output
        model_outputs = style_outputs + [content_output]

        # Create model that outputs both style and content features
        self.model = tf.keras.models.Model(vgg.input, model_outputs)

    @staticmethod
    def gram_matrix(input_tensor):
        '''Calculates the Gram matrix for the given input tensor'''
        if not isinstance(input_tensor, (tf.Tensor, tf.Variable)) or len(input_tensor.shape) != 4:
            raise TypeError('input_layer must be a tensor of rank 4')

        # Reshape and calculate the Gram matrix
        _, h, w, c = input_tensor.shape
        features = tf.reshape(input_tensor, [-1, c])
        gram = tf.matmul(features, features, transpose_a=True)
        return gram / tf.cast(h * w, tf.float32)

    def generate_features(self):
        '''Extract features used to calculate neural style cost'''
        vgg_preprocess = tf.keras.applications.vgg19.preprocess_input
        style_image = vgg_preprocess(self.style_image * 255)
        content_image = vgg_preprocess(self.content_image * 255)

        outputs_style = self.model(style_image)[:-1]
        output_content = self.model(content_image)[-1]

        self.gram_style_features = [self.gram_matrix(output) for output in outputs_style]
        self.content_feature = output_content

    def layer_style_cost(self, style_output, gram_target):
        '''Calculates the style cost for a single layer'''
        if not isinstance(style_output, (tf.Tensor, tf.Variable)) or len(style_output.shape) != 4:
            raise TypeError('style_output must be a tensor of rank 4')

        _, _, _, c = style_output.shape

        if not isinstance(gram_target, (tf.Tensor, tf.Variable)) or gram_target.shape != (1, c, c):
            raise TypeError('gram_target must be a tensor of shape [1, {}, {}]'.format(c, c))

        gram_style_output = self.gram_matrix(style_output)

        return tf.reduce_mean(tf.square(gram_style_output - gram_target))



