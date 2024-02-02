#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

# Set the mean and covariance matrix
mean = [69, 0]
cov = [[15, 8], [8, 15]]

# Generate random samples based on the mean and covariance
np.random.seed(5)
x, y = np.random.multivariate_normal(mean, cov, 2000).T

# Calculate mean values for x and y
y_mean = np.mean(y)
print("Mean of y =", y_mean)
x_mean = np.mean(x)
print("Mean of x:", x_mean)

# Offset y values
y += 180

# Scatter plot
plt.scatter(x, y, c='m')
plt.title("Men's Height vs Weight")
plt.xlabel('Height (in)')
plt.ylabel('Weight (lbs)')
plt.show()
