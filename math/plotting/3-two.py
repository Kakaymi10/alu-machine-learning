#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

# Generate data for radioactive decay of C-14 and Ra-226
x = np.arange(0, 21000, 1000)
r = np.log(0.5)
t1 = 5730
t2 = 1600
y1 = np.exp((r / t1) * x)
y2 = np.exp((r / t2) * x)

# Plotting
plt.title("Exponential Decay of Radioactive Elements")
plt.plot(x, y1, c='r', linestyle='--', label='C-14')
plt.plot(x, y2, c='g', label='Ra-226')
plt.xlim(left=0, right=20000)
plt.ylim(bottom=0, top=1)
plt.xlabel('Time (years)')
plt.ylabel('Fraction Remaining')
plt.legend()  # Add legend
plt.show()
