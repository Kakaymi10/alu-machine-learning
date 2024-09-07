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
                 beta=1,
                 var=10):
        '''
        Creates an instance of the NST class

        Args:
            style_image: image np array used as style reference
            content_image: image np.array used as content reference
            alpha: the weight of the content cost
            beta: the weight of the style cost
            var: the weight of the variational cost

        Returns:
            An instance of the NST Class
        '''

        # Check style_image type and shape
        if not (isinstance(style_image, np.ndarray) and
                len(np.shape(style_image)) == 3 and
                np.shape(style_image)[2] == 3):
            raise TypeError(
                'style_image must be a numpy.ndarray with shape (h, w, 3)')

        # Check content_image type and shape
        if not (isinstance(content_image, np.ndarray) and
                len(np.shape(content_image)) == 3 and
                np.shape(content_image)[2] == 3):
            raise TypeError(
                'content_image must be a numpy.ndarray with shape (h, w, 3)')

        # Ensure alpha, beta, and var are non-negative numbers
        if (type(alpha) not in [float, int]) or alpha < 0:
            raise TypeError('alpha must be a non-negative number')

        if (type(beta) not in [float, int]) or beta < 0:
            raise TypeError('beta must be a non-negative number')

        if (type(var) not in [float, int]) or var < 0:
            raise TypeError('var must be a non-negative number')

        # Disable lazy execution tf v1.12
        tf.enable_eager_execution()

        # Instance attributes
        self.style_image = self.scale_image(style_image)
        self.content_image = self.scale_image(content_image)
        self.alpha = alpha
        self.beta = beta
        self.var = var

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
        if not (isinstance(image, np.ndarray) and
                len(np.shape(image)) == 3 and
                np.shape(image)[2] == 3):
            raise TypeError('image must be a numpy.ndarray with shape (h, w, 3)')

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

        # Load VGG model
        vgg_model = tf.keras.applications.VGG19(include_top=False,
                                                weights='imagenet')

        # MaxPooling2D - AveragePooling2D
        vgg_model.save('base')
        custom_objects = {'MaxPooling2D': tf.keras.layers.AveragePooling2D}
        vgg = tf.keras.models.load_model('base', custom_objects=custom_objects)

        style_outputs = [vgg.get_layer(name).output
                         for name in self.style_layers]
        content_outputs = [vgg.get_layer(self.content_layer).output]
        model_outputs = style_outputs + content_outputs

        model = tf.keras.models.Model(vgg.input, model_outputs,
                                      name="model")

        # Freeze weights
        model.trainable = False

        self.model = model

    @staticmethod
    def gram_matrix(input_layer):
        '''
        Calculates gram matrices

        Args:
            input_layer: an instance of tf.Tensor or
            tf.Variable of shape (1, h, w, c) containing
            the output whose gram matrix should be calculated

        Returns:
            A tf.Tensor of shape (1, c, c) containing
            the gram matrix of input layer
        '''

        # Run checks
        if not (isinstance(input_layer, tf.Tensor) or
                isinstance(input_layer, tf.Variable)) or \
                len(input_layer.shape) != 4:
            raise TypeError('input_layer must be a tensor of rank 4')

        channels = int(input_layer.shape[-1])
        a = tf.reshape(input_layer, [1, -1, channels])
        n = tf.shape(a)[1]

        gram = tf.matmul(a, a, transpose_a=True)
        return gram / tf.cast(n, tf.float32)

    def layer_style_cost(self, style_output, gram_target):
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
                isinstance(style_output, tf.Variable)) or \
                len(style_output.shape) != 4:
            raise TypeError('style_output must be a tensor of rank 4')

        _, _, _, c = style_output.shape

        if not (isinstance(gram_target, tf.Tensor) or
                isinstance(gram_target, tf.Variable)) or \
                gram_target.shape != (1, c, c):
            raise TypeError('gram_target must be a tensor of shape '
                            '[1, {}, {}]'.format(c, c))

        gram_style_output = self.gram_matrix(style_output)

        return tf.reduce_mean(tf.square(gram_style_output - gram_target))

    def style_cost(self, style_outputs):
        '''
        Calculates the style cost for the generated image

        Args:
            style_outputs: list of tf.Tensor style outputs

        Return:
            The style cost
        '''
        length = len(self.style_layers)

        if not isinstance(style_outputs, list) or len(style_outputs) != length:
            raise TypeError('style_outputs must be a list with a length '
                            'of {}'.format(length))

        style_cost = 0.0
        weight_per_style = 1.0 / length

        for i in range(length):
            style_cost += weight_per_style * \
                self.layer_style_cost(style_outputs[i],
                                      self.gram_style_features[i])

        return style_cost

    # Content Cost
    def content_cost(self, content_output):
        '''
        Calculates the content cost for the generated image

        Args:
            content_output: a tf.Tensor containing the content output

        Returns:
            Content cost
        '''
        s = self.content_feature.shape

        if not (isinstance(content_output, tf.Tensor) or
                isinstance(content_output, tf.Variable)) or \
                content_output.shape != s:
            raise TypeError('content_output must be a tensor of shape '
                            '{}'.format(s))

        return tf.reduce_mean(tf.square(content_output - self.content_feature))

    # Variational Cost
    @staticmethod
    def variational_cost(generated_image):
        '''
        Calculates the variational cost for the generated image

        Args:
            generated_image: a tf.Tensor of shape (1, nh, nw, 3)
            containing the generated image

        Returns:
            The variational cost
        '''
        if not (isinstance(generated_image, tf.Tensor) or
                isinstance(generated_image, tf.Variable)) or \
                len(generated_image.shape) != 4:
            raise TypeError('generated_image must be a tensor of rank 4')

        nh, nw, _ = generated_image.shape[1:-1]

        # Create filters for gradient calculation
        filter_x = tf.constant([[[[-1, 1]]]], dtype=tf.float32)
        filter_y = tf.constant([[[[-1], [1]]]], dtype=tf.float32)

        # Calculate gradients
        grad_x = tf.nn.conv2d(generated_image, filter_x,
                              strides=[1, 1, 1, 1], padding='VALID')
        grad_y = tf.nn.conv2d(generated_image, filter_y,
                              strides=[1, 1, 1, 1], padding='VALID')

        # Calculate variational cost
        grad_x_sq = tf.square(grad_x)
        grad_y_sq = tf.square(grad_y)

        return tf.reduce_mean(grad_x_sq + grad_y_sq)

    def total_cost(self, generated_image):
        '''
        Calculates the total cost of the generated image

        Args:
            generated_image: the generated image

        Returns:
            total cost
        '''
        # Extract the style, content and variational costs
        model_outputs = self.model(generated_image)

        style_outputs = model_outputs[:-1]
        content_output = model_outputs[-1]

        style_cost = self.style_cost(style_outputs)
        content_cost = self.content_cost(content_output)
        variational_cost = self.variational_cost(generated_image)

        total_cost = (self.alpha * content_cost +
                      self.beta * style_cost +
                      self.var * variational_cost)

        return total_cost

    def generate_image(self, iterations=100, step=None):
        '''
        Generate an image that minimizes the total cost

        Args:
            iterations: number of iterations
            step: If not None, will print the cost every step iterations

        Returns:
            The generated image
        '''
        # Initialize generated image
        generated_image = np.copy(self.content_image)
        generated_image = tf.Variable(generated_image, dtype=tf.float32)

        # Optimizer
        optimizer = tf.keras.optimizers.Adam(learning_rate=10.0)

        for i in range(iterations):
            with tf.GradientTape() as tape:
                cost = self.total_cost(generated_image)
                grads = tape.gradient(cost, generated_image)
                optimizer.apply_gradients([(grads, generated_image)])
                if step and i % step == 0:
                    print("Iteration {}: Cost = {}".format(i, cost.numpy()))

        return generated_image.numpy()
