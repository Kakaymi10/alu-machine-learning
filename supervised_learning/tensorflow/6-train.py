#!/usr/bin/env python3
"""Train a neural network model using TensorFlow."""


import tensorflow as tf
from importlib import import_module


# Import required functions
calculate_accuracy = import_module('3-calculate_accuracy').calculate_accuracy
calculate_loss = import_module('4-calculate_loss').calculate_loss
create_placeholders = import_module('0-create_placeholders').create_placeholders
create_train_op = import_module('5-create_train_op').create_train_op
forward_prop = import_module('2-forward_prop').forward_prop

def train(X_train, Y_train, X_valid, Y_valid, layer_sizes, activations, alpha, iterations, save_path="/tmp/model.ckpt"):
    """
    Trains a neural network model and
    return path where the model is saved.
    """
    
    # Create placeholders for input data and labels
    x, y = create_placeholders(X_train.shape[1], Y_train.shape[1])
    tf.add_to_collection('x', x)
    tf.add_to_collection('y', y)
    
    # Forward propagation to get predictions
    y_pred = forward_prop(x, layer_sizes, activations)
    tf.add_to_collection('y_pred', y_pred)
    
    # Calculate accuracy and loss
    accuracy = calculate_accuracy(y, y_pred)
    tf.add_to_collection('accuracy', accuracy)
    loss = calculate_loss(y, y_pred)
    tf.add_to_collection('loss', loss)
    
    # Create the training operation
    train_op = create_train_op(loss, alpha)
    tf.add_to_collection('train_op', train_op)
    
    # Initialize global variables
    init = tf.global_variables_initializer()
    saver = tf.train.Saver()
    
    with tf.Session() as sess:
        sess.run(init)
        
        # Training loop
        for i in range(iterations + 1):
            # Calculate training and validation metrics
            cost_t = sess.run(loss, feed_dict={x: X_train, y: Y_train})
            acc_t = sess.run(accuracy, feed_dict={x: X_train, y: Y_train})
            cost_v = sess.run(loss, feed_dict={x: X_valid, y: Y_valid})
            acc_v = sess.run(accuracy, feed_dict={x: X_valid, y: Y_valid})
            
            # Print metrics every 100 iterations and at the end
            if i % 100 == 0 or i == iterations:
                print(f"After {i} iterations:")
                print(f"\tTraining Cost: {cost_t}")
                print(f"\tTraining Accuracy: {acc_t}")
                print(f"\tValidation Cost: {cost_v}")
                print(f"\tValidation Accuracy: {acc_v}")
            
            # Run the training operation
            if i < iterations:
                sess.run(train_op, feed_dict={x: X_train, y: Y_train})
        
        # Save the trained model
        return saver.save(sess, save_path)
