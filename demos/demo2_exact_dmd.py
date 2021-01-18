"""
Exact-DMD on minimum jerk trajectories.
"""

import numpy as np
import jax.numpy as jnp
import jax
jax.config.update('jax_platform_name', 'cpu')
import matplotlib.pyplot as plt

from datasets import *
from models.dmd_models import *
from utilities.utils import data2snapshots, snapshots2data


def demo():
    """
    This demo shows an example of the exact dynamic mode decomposition on the minimum_jerk_trajectories data.
    :return: None
    """
    np.random.seed(5)
    t_steps = jnp.arange(0, 1, 0.01)
    samples = 1
    time_delay = 1
    axis = 1
    sigma = 0.001
    trunc_svd = -1
    dmd_method = "standard"

    x_init = jnp.deg2rad(jnp.array([[0.], [50.]]))
    x_final = jnp.deg2rad(jnp.array([[50.], [-20.]]))

    data = MinimumJerk(x_init, x_final, t_steps, s_size=samples, sigma=sigma)
    x0, x1 = data2snapshots(data.transform, t_delay=time_delay, axis=axis)

    if dmd_method == "standard":
        dmd = StandardDMD()
    elif dmd_method == "exact":
        dmd = ExactDMD()
    elif dmd_method == "tls":
        dmd = TLSDMD()
    elif dmd_method == "fb":
        dmd = FBDMD()
    else:
        raise NotImplementedError(
            'DMD method {} not implement'.format(dmd_method))

    dmd.fit(x0, x1, trunc_svd=trunc_svd, t_delay=time_delay, axis=axis)
    out = jnp.real(dmd.predict(t_steps))

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

    ax11.plot(t_steps, out[:idx, :].T, '--')
    ax12.plot(t_steps, out[idx:2*idx, :].T, '--')
    ax13.plot(t_steps, out[2*idx:3*idx, :].T, '--')

    plt.suptitle("Minimum Jerk Trajectories")
    plt.show()


if __name__ == '__main__':
    demo()