#!/usr/bin/env python3
"""Evaluate a trained neural network model using TensorFlow."""


import tensorflow as tf


def evaluate(X, Y, save_path):
    """
    Evaluates a trained model and
    returns a tuple of predictions, accuracy, and loss.
    """

    with tf.Session() as sess:
        # Restore the saved model
        saver = tf.train.import_meta_graph(save_path + '.meta')
        saver.restore(sess, save_path)

        # Retrieve tensors from the collection
        x = tf.get_collection('x')[0]
        y = tf.get_collection('y')[0]
        y_pred = tf.get_collection('y_pred')[0]
        accuracy = tf.get_collection('accuracy')[0]
        loss = tf.get_collection('loss')[0]

        # Evaluate the model on the given data
        pred = sess.run(y_pred, feed_dict={x: X, y: Y})
        acc = sess.run(accuracy, feed_dict={x: X, y: Y})
        ls = sess.run(loss, feed_dict={x: X, y: Y})

    return pred, acc, ls
