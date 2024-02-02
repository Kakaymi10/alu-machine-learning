#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

def generate_fruit_data():
    # Set seed for reproducibility
    np.random.seed(5)

    # Generate random fruit data
    return np.random.randint(0, 20, (4, 3))

def plot_fruit_data(fruit_data):
    # Print the generated fruit data
    print(fruit_data)

if __name__ == "__main__":
    fruit_data = generate_fruit_data()
    plot_fruit_data(fruit_data)
