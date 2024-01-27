#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

x = np.arange(0, 11)
y = x ** 3

plt.plot(x, y, c="r")
# set left and right limits
plt.xlim(left=min(x), right=max(x))
plt.show()
