#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

# Generate data for C-14 decay
x = np.arange(0, 28651, 5730)
r = np.log(0.5)
t = 5730
y = np.exp((r / t) * x)

# Plotting
plt.title("Exponential Decay of C-14")
plt.plot(x, y)
plt.xlim(left=0, right=28650)
plt.xlabel('Time (years)')
plt.ylabel('Fraction Remaining')
plt.yscale('log')  # Set y-axis to logarithmic scale
plt.show()
