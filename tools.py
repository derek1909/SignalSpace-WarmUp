import torch
import torch.nn as nn

import matplotlib.pyplot as plt
import numpy as np

def GenerateDataset(a=2, b=0.7, num_noisy_samples=30, noise_sigma=0.2, PLOT=True):
    # Range of theta values
    t = np.linspace(0, 3 * np.pi, 360)  # From 0 to 5*pi

    # Single curve
    theta = t
    r = a + b * t

    # twin curve
    theta = np.hstack((theta, theta))
    r = np.hstack((r, -r))

    # Generate original curve
    xy = np.array([r * np.cos(theta), r * np.sin(theta)]).T

    # Initialize arrays to hold the noisy data
    xy_noisy = np.tile(xy, [num_noisy_samples, 1])  # Repeat theta for each set of noisy points

    # Generate noisy data points
    noise = np.random.normal(0, noise_sigma, xy_noisy.shape)
    xy_noisy += noise

    if PLOT:
        # Plotting in polar coordinates
        plt.figure(figsize=[8, 8])

        # Original curve
        ax = plt.subplot(111)
        ax.plot(xy[:,0], xy[:,1], label='Original Curve', color='blue')

        # Noisy data points
        ax = plt.subplot(111)
        ax.scatter(xy_noisy[:, 0], xy_noisy[:, 1], label='Noisy Data', color='red', alpha=0.1)

        # Setting the title and labels
        ax.set_title("Original Curve and Noisy Data Points: r(θ) = a + bθ")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.legend()

        # Show the plot
        plt.show()

    return xy_noisy

