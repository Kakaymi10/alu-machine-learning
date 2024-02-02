#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

# Set seed for reproducibility
np.random.seed(5)

# Generate random student grades with a normal distribution
student_grades = np.random.normal(68, 15, 50)
print(student_grades)

# Plotting a histogram
plt.hist(student_grades, bins=10, range=(0, 100), edgecolor='black')
plt.xlabel('Grades')
plt.ylabel('Number of Students')
plt.title('Distribution of Student Grades - Project A')
plt.xlim(0, 100)
plt.ylim(0, 30)
plt.show()
