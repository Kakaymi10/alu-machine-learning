#!/usr/bin/env python3
'''Create a class NST that performs tasks for neural style transfer'''


import numpy as np
import tensorflow as tf


class NST:
    '''
    This class performs neural style transfer
    '''

    # Declare public class attributes
    # Accessible from outside the class
    style_layers = ['block1_conv1',
                    'block2_conv1',
                    'block3_conv1',
                    'block4_conv1',
                    'block5_conv1']

    content_layer = 'block5_conv2'

    # Class constructor
    def __init__(self,
                 style_image,
                 content_image,
                 alpha=1e4,
                 beta=1):
        '''
        Creates an instance of the NST class

        Args:
            style_image: image np array used as style reference
            content_image: image np.array used as content reference
            alpha: the weight of the content cost
            beta: the weight of the style cost

        Returns:
            An instance of the NST Class
        '''

        # Check style_image type and shape
        if not (isinstance(
            style_image, np.ndarray) and len(
                np.shape(style_image)) == 3 and np.shape(
                    style_image)[2] == 3):
            raise TypeError(
                'style_image must be a numpy.ndarray with shape (h, w, 3)')

        # Check content_image type and shape
        if not (isinstance(
            content_image, np.ndarray) and len(
                np.shape(content_image)) == 3 and np.shape(
                    content_image)[2] == 3):
            raise TypeError(
                'content_image must be a numpy.ndarray with shape (h, w, 3)')

        # Ensure alpha and beta are non-negative numbers
        if (
            type(alpha) is not float and type(
                alpha) is not int) or alpha < 0:
            raise TypeError('alpha must be a non-negative number')

        if (
            type(beta) is not float and type(
                beta) is not int) or beta < 0:
            raise TypeError('beta must be a non-negative number')

        # disable lazy execution tf v1.12
        tf.enable_eager_execution()

        # Instance attributes
        self.style_image = self.scale_image(style_image)
        self.content_image = self.scale_image(content_image)
        self.alpha = alpha
        self.beta = beta

        self.load_model()
        self.generate_features()

    @staticmethod
    def scale_image(image):
        '''
        Rescales an image such that
        values are between 0 and 1 and largest
        side is 512 pixels

        Args:
            image - a np.ndarray with shape (h, w, 3)

        Returns:
            the scaled image
        '''
        if not (
            isinstance(image, np.ndarray) and len(
                np.shape(image)) == 3 and np.shape(
                    image)[2] == 3):
            raise TypeError(
                'image must be a numpy.ndarray with shape (h, w, 3)')

        h, w, _ = image.shape

        if h > w:
            h_new = 512
            w_new = int(w * (512 / h))
        else:
            w_new = 512
            h_new = int(h * (512 / w))

        resized = tf.image.resize_bicubic(
            np.expand_dims(image, axis=0), size=(h_new, w_new))

        rescaled = resized / 255
        rescaled = tf.clip_by_value(rescaled, 0, 1)

        return rescaled

    # Public Instance Method
    def load_model(self):
        '''
        Creates the model used to calculate the loss
        '''

        # load vgg model
        vgg_model = tf.keras.applications.VGG19(
            include_top=False, weights='imagenet')

        # MaxPooling2D - AveragePooling 2D
        vgg_model.save('base')
        custom_objects = {'MaxPooling2D': tf.keras.layers.AveragePooling2D}
        vgg = tf.keras.models.load_model(
            'base', custom_objects=custom_objects)

        style_outputs = [
            vgg.get_layer(name).output for name in self.style_layers]
        content_outputs = [
            vgg.get_layer(self.content_layer).output]
        model_outputs = style_outputs + content_outputs

        model = tf.keras.models.Model(
            vgg.input, model_outputs, name="model")

        # Freeze weights
        model.trainable = False

        self.model = model

    @staticmethod
    def gram_matrix(input_layer):
        '''
        Calculates gram matrices

        Args:
            input_layer: an instance of tf.tensor or
            tf.Variable of shape (1, h, w, c) containing
            the output whose gram matrix should be calculated

        Returns:
            A tf.Tensor of shape (1, c, c) containing
            the gram matrix of input layer
        '''

        # Run checks
        if not (isinstance(input_layer, tf.Tensor) or
                isinstance(input_layer, tf.Variable)) or len(
                    input_layer.shape) != 4:
            raise TypeError('input_layer must be a tensor of rank 4')

        channels = int(input_layer.shape[-1])
        a = tf.reshape(input_layer, [1, -1, channels])
        n = tf.shape(a)[1]

        gram = tf.matmul(a, a, transpose_a=True)
        return gram / tf.cast(n, tf.float32)

    def generate_features(self):
        '''
        Extract features used to calculate neural style cost
        '''
        vgg_model = tf.keras.applications.vgg19

        preprocess_style = vgg_model.preprocess_input(
            self.style_image * 255)
        preprocess_content = vgg_model.preprocess_input(
            self.content_image * 255)

        style_features = self.model(preprocess_style)[:-1]
        content_feature = self.model(preprocess_content)[-1]

        gram_style_features = []
        for feature in style_features:
            gram_style_features.append(self.gram_matrix(feature))

        self.gram_style_features = gram_style_features
        self.content_feature = content_feature

    def layer_style_cost(self,
                         style_output,
                         gram_target):
        '''
        Calculates the style cost for a single layer

        Args:
            - style_output: (1, h, w, c) tf.Tensor containing
            the style output of the generated image

            - gram_target: (1, c, c) tf.Tensor of the target
            style output for that layer

        Returns:
            Layer's style cost
        '''
        if not (isinstance(style_output, tf.Tensor) or
                isinstance(style_output, tf.Variable)) or len(
                    style_output.shape) != 4:
            raise TypeError('style_output must be a tensor of rank 4')

        _, _, _, c = style_output.shape

        if not (isinstance(gram_target, tf.Tensor) or
                isinstance(gram_target,
                           tf.Variable)) or gram_target.shape != (1, c, c):
            raise TypeError(
                'gram_target must be a tensor of shape [1, {}, {}]'
                .format(c, c))

        gram_style_output = self.gram_matrix(style_output)

        return tf.reduce_mean(
            tf.square(gram_style_output - gram_target))

    def style_cost(self, style_outputs):
        '''
        Calculates the style cost for the generated image

        Args:
            style_outputs: list of tf.Tensor style outputs

        Return:
            The style cost
        '''
        length = len(self.style_layers)

        if not isinstance(style_outputs, list) or len(
                style_outputs) != length:
            raise TypeError(
                'style_outputs must be a list with a length of {}'.format(
                    length))

        style_cost = 0.0
        weight_per_style = 1.0 / length

        for i in range(length):
            style_cost += weight_per_style * self.layer_style_cost(
                style_outputs[i], self.gram_style_features[i]
            )

        return style_cost

    # Content Cost
    def content_cost(self, content_output):
        '''
        Calculcates the content cost for the generated image

        Args:
            content_output: a tf.Tensor containing the content output

        Returns:
            Content cost
        '''
        s = self.content_feature.shape

        if not (isinstance(content_output, tf.Tensor) or
                isinstance(content_output,
                           tf.Variable)) or content_output.shape != s:
            raise TypeError(
                'content_output must be a tensor of shape {}'.format(s))

        return tf.reduce_mean(
            tf.square(content_output - self.content_feature)
        )

    # Calculate the total cost
    def total_cost(self, generated_image):
        '''
        Calculates the total cost for the generated image

        Args:
            generated_image: a tf.Tensor of shape (1, nh, nw, 3)

        Returns:
            J, J_content, J_style
        '''
        s = self.content_image.shape

        if not (isinstance(generated_image, tf.Tensor) or
                isinstance(generated_image,
                           tf.Variable)) or generated_image.shape != s:
            raise TypeError(
                'generated_image must be a tensor of shape {}'.format(s)
            )

        vgg19 = tf.keras.applications.vgg19
        preprocessed_gen_image = vgg19.preprocess_input(
            generated_image * 255)
        outputs = self.model(preprocessed_gen_image)

        content_output = outputs[-1]
        style_outputs = outputs[:-1]

        J_content = self.content_cost(content_output)
        J_style = self.style_cost(style_outputs)

        J = self.alpha * J_content + self.beta * J_style

        return J, J_content, J_style

    def compute_grads(self, generated_image):
        '''
        Calculates the gradients for the generated image in TensorFlow 1.1x

        Args:
            generated_image: a tf.Tensor of shape (1, nh, nw, 3)
            alpha: weight for the content cost
            beta: weight for the style cost

        Returns:
            gradients: tf.Tensor containing the
            gradients for the generated image
            J_total: total cost for the generated image
            J_content: content cost for the generated image
            J_style: style cost for the generated image
        '''
        s = self.content_image.shape
        if not (isinstance(generated_image,
                           tf.Tensor) or isinstance(
                               generated_image,
                               tf.Variable)) or generated_image.shape != s:
            raise TypeError(
                'generated_image must be a tensor of shape {}'.format(s)
            )

        with tf.GradientTape() as tape:
            J_total, J_content, J_style = self.total_cost(generated_image)

        gradients = tape.gradient(J_total, generated_image)

        return gradients, J_total, J_content, J_style
