"""
Exact-DMD on minimum jerk trajectories.
"""

import numpy as np
import torch as tr
import matplotlib.pyplot as plt

from data.minimum_jerk_trajectories import MinimumJerk
from dmd.exact_dmd import ExactDMD


def demo():
    """
    This demo shows an example of the exact dynamic mode decomposition on the minimum_jerk_trajectories data.
    :return: None
    """
    tr.manual_seed(0)
    t_steps = tr.arange(0, 1, 0.01)
    trajectories = 1

    x_init = tr.deg2rad(tr.tensor([[0.], [50.]]))
    x_final = tr.deg2rad(tr.tensor([[50.], [-20.]]))

    data_generator = MinimumJerk()
    data = data_generator.get_data(t_steps, x_init, x_final, trajectories=trajectories, sigma=0.1)

    exact_dmd = ExactDMD()
    X0 = data[:, :, :-1]
    X1 = data[:, :, 1:]
    print(tr.cat((X0, X1), dim=0).shape)  # TODO possible error
    exact_dmd.fit(tr.cat((X0, X1), dim=0))
    out = exact_dmd.predict(t_steps)

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
        ax11.plot(t_steps, np.real(out[idx, :]).T, 'g--')
        ax11.plot(t_steps, data[:, idx, :].T, 'b', alpha=.5)
        ax12.plot(t_steps, np.real(out[n + idx, :]).T, 'g--')
        ax12.plot(t_steps, data[:, n + idx, :].T, 'b', alpha=.5)
        ax13.plot(t_steps, np.real(out[2*n + idx, :]).T, 'g--')
        ax13.plot(t_steps, data[:, 2*n + idx, :].T, 'b', alpha=.5)

    plt.suptitle("Minimum Jerk Trajectories")
    plt.show()


if __name__ == '__main__':
    demo()