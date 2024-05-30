# Regularization: Making Models Behave, Because Overfitting is so Last Season

## We...

## We wondered why our models acted like toddlers on a sugar high. 

## We realized they were overfitting faster than we overfit our jeans after Thanksgiving dinner.

## We decided it was time to bring in the discipline—no more overindulgence in fitting the training data!

## We laughed, we cried, and then we turned to regularization.

# What the Heck is Regularization?

Regularization is like sending your model to a yoga retreat—it's all about balance. It prevents the model from getting too attached to the training data, so it performs better on new, unseen data.

### Types of Regularization:

1. **L1 and L2 Regularization**:
   - **L1 Regularization (Lasso)**: Forces some weights to become zero, which can make the model simpler and more interpretable.
   - **L2 Regularization (Ridge)**: Encourages smaller weights overall, but none go completely to zero. It’s like a gentle reminder to the model to keep its parameters in check.

2. **Dropout**: Imagine your neurons playing a game of hide and seek. During training, some neurons randomly drop out, preventing them from co-depending too much on each other.

3. **Early Stopping**: This is like telling your model, "Okay, that's enough!" If the model's performance on validation data stops improving, training is halted to prevent overfitting.

4. **Data Augmentation**: Think of this as giving your model new glasses. By artificially expanding the training data through transformations like rotations or flips, your model sees the world in more diverse ways.

## Resources
Dive into these resources to master the art of regularization:

### Read or Watch:
- **Regularization (mathematics)**
- **An Overview of Regularization Techniques in Deep Learning**
- **L2 Regularization and Back-Propagation**
- **Intuitions on L1 and L2 Regularisation**
- **Analysis of Dropout**
- **Early Stopping**
- **How to use early stopping properly for training deep neural network?**
- **Data Augmentation | How to use Deep Learning when you have Limited Data**
- **deeplearning.ai videos**: (Recommended at 1.5x - 2x speed for extra fun)
  - Regularization
  - Why Regularization Reduces Overfitting
  - Dropout Regularization
  - Understanding Dropout
  - Other Regularization Methods

### References:
- `numpy.linalg.norm`
- `numpy.random.binomial`
- `tf.contrib.layers.l2_regularizer`
- `tf.layers.Dense#kernel_regularizer`
- `tf.losses.get_regularization_loss`
- `tf.layers.Dropout`
- **Dropout: A Simple Way to Prevent Neural Networks from Overfitting**
- **Early Stopping - but when?**
- **L2 Regularization versus Batch and Weight Normalization**

## Learning Objectives
By the end of this journey, you’ll be able to explain to anyone, even your grandma:

- **What is regularization?** What is its purpose?
- **What are L1 and L2 regularization?** What is the difference between them?
- **What is dropout?**
- **What is early stopping?**
- **What is data augmentation?**
- **How do you implement these regularization methods in Numpy and TensorFlow?**
- **What are the pros and cons of each method?**

