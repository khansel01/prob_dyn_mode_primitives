"""
This demo shows how to generate and receive data. Furthermore it shows how the data looks like.
This demo uses minimal jerk data from data/minimum_jerk_trajectories.
"""

import numpy as np
import torch as tr
import matplotlib.pyplot as plt

from data.minimum_jerk_trajectories import MinimumJerk


def demo():
    t_steps = tr.arange(0, 1, 0.01)
    trajectories = 10

    x_init = tr.tensor([[np.radians(0)], [np.radians(0)]])
    x_final = tr.tensor([[np.radians(50)], [np.radians(-20)]])

    data_generator = MinimumJerk()
    data = data_generator.get_data(t_steps, x_init, x_final, trajectories=trajectories, sigma=0.1)

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

    n = x_init.shape[0]
    for idx in range(n):
        ax11.plot(t_steps, np.real(data[:, idx, :]).T)
        ax12.plot(t_steps, np.real(data[:, n + idx, :]).T)
        ax13.plot(t_steps, np.real(data[:, 2*n + idx, :]).T)

    plt.suptitle("Minimum Jerk Trajectories")
    plt.show()


if __name__ == '__main__':
    demo()
