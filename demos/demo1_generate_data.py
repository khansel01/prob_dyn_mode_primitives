"""
Demo for python script data/minimum_jerk_trajectories.
"""

import numpy as np
import matplotlib.pyplot as plt

from datasets.minimum_jerk_trajectories import MinimumJerk


def demo():
    """
    This demo shows how to generate and receive data. Furthermore it shows how the data looks like.
    This demo uses minimal jerk data from data/minimum_jerk_trajectories.
    :return: None
    """
    t_steps = np.arange(0, 1, 0.01)
    samples = 10
    sigma = 0.1

    x_init = np.deg2rad(np.array([[0.], [0.]]))
    x_final = np.deg2rad(np.array([[50.], [-20.]]))

    data = MinimumJerk(x_init, x_final, t_steps, s_size=samples, sigma=sigma)

    # Visualize generated data
    fig = plt.figure()
    grid = fig.add_gridspec(3, 1)
    ax11 = fig.add_subplot(grid[0, 0])
    ax11.set_title("Position")
    ax11.set_xlabel("Time")
    ax11.set_ylabel("$\Theta$")

    ax12 = fig.add_subplot(grid[1, 0])
    ax12.set_title("Velocity")
    ax12.set_xlabel("Time")
    ax12.set_ylabel("$\dot{\Theta}$")

    ax13 = fig.add_subplot(grid[2, 0])
    ax13.set_title("Acceleration")
    ax13.set_xlabel("Time")
    ax13.set_ylabel("$\ddot{\Theta}$")

    idx = x_init.shape[0]
    for d in data.transform:
        ax11.plot(t_steps, d[:idx, :].T)
        ax12.plot(t_steps, d[idx:2*idx, :].T)
        ax13.plot(t_steps, d[2*idx:3*idx, :].T)

    plt.suptitle("Minimum Jerk Trajectories")
    plt.show()


if __name__ == '__main__':
    demo()
