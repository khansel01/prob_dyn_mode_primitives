import matplotlib.pyplot as plt
import argparse

from datasets import MinimumJerk
from utilities.utils import data2snapshots, pos_def
from utilities.prng_handler import PRNGHandler
from utilities.kernel_lib import *
from models.gp_models.gp_dmd import GPDMD

parser = argparse.ArgumentParser(description="Test")
parser.add_argument("--device", default="cpu", type=str,
                        help="Specify the device on which Jax works. Possible devices are cpu or gpu (if available).")
parser.add_argument("--x64", default=True, type=bool,
                        help="Defines the float type used by Jax.")

parser.add_argument("--kernel", default="RBF", type=str,
                    help="Specifies the Kernel used in KernelDMD. Possible Methods are Polynomial and RBF")

parser.add_argument("--seed", default=11, type=int,
                        help="Defines the seed of the pseudo random number generator.")
parser.add_argument("--sigma", default=0.1, type=float,
                    help="Set the variability in the sampled data.")
parser.add_argument("--samples", default=1, type=int,
                        help="Set the number of sampled trajectories.")
parser.add_argument("--time_delay", default=1, type=int,
                        help="Set an integer to define the time shift used to create delayed coordinates.")
parser.add_argument("--axis", default=1, type=int,
                        help="Set the axis along which the trajectories will be concatenated.")

parser.add_argument("--latent_dim", default=5, type=int,
                        help="Dimensionality of the latent space.")
parser.add_argument("--iterations", default=10000, type=int,
                    help="Specifies the iterations of the Optimizer.")


def demo():
    """
    This demo shows an example of the exact dynamic mode decomposition on the minimum_jerk_trajectories data.
    :return: None
    """
    args = parser.parse_args()

    jax.config.update('jax_platform_name', args.device)
    jax.config.update('jax_enable_x64', args.x64)

    t_steps = jnp.arange(0, 1, 0.01)

    prng_handler = PRNGHandler(seed=args.seed)

    # Generate Data
    x_init = jnp.deg2rad(jnp.array([[-10.], [50.]]))
    x_final = jnp.deg2rad(jnp.array([[50.], [-20.]]))
    data = MinimumJerk(x_init, x_final, t_steps, s_size=args.samples, sigma=args.sigma, prng_handler=prng_handler)
    x0, x1 = data2snapshots(data.transform, t_delay=args.time_delay, axis=args.axis)
    x = jnp.hstack((x0, x1[:, -1].reshape(-1, 1)))

    kernel_fun = eval(f'{args.kernel}Kernel')

    gp_dmd = GPDMD(iterations=args.iterations, latent_dim=args.latent_dim,
                  prng_handler=prng_handler, kernel=kernel_fun())

    gp_dmd.fit(x)

    z_latent = gp_dmd.predict(t_steps).T

    labels_idx = [0, 33, 66, 99]
    labels_x =  x.T[labels_idx, :]
    _samples = gp_dmd.get_sample(labels_x, labels_idx, t_steps, 10)

    # Visualize Data
    fig = plt.figure()
    grid = fig.add_gridspec(4, 1)

    ax11 = fig.add_subplot(grid[0, 0])
    ax11.set_title(f"Target Data")
    ax11.set_xlabel("Time")
    ax11.set_ylabel("$\Theta$")
    plt.gca().set_prop_cycle(None)
    ax11.plot(t_steps, data.transform[0].T)

    ax12 = fig.add_subplot(grid[1, 0])
    ax12.set_title(f"Samples")
    ax12.set_xlabel("Time")
    ax12.set_ylabel("$\Theta$")
    for s in _samples:
        plt.gca().set_prop_cycle(None)
        ax12.plot(t_steps, jnp.real(s.T))

    ax14 = fig.add_subplot(grid[2, 0])
    ax14.set_title(f"Latent_dim")
    ax14.set_xlabel("Time")
    ax14.set_ylabel("$\Theta$")
    ax14.plot(t_steps, jnp.real(z_latent), '--')

    ax15 = fig.add_subplot(grid[3, 0])
    ax15.set_title("Log_Likelihood")
    ax15.set_xlabel("Sample")
    ax15.set_ylabel("$log_likelihood value$")
    ax15.plot(jnp.arange(len(gp_dmd.ll_values)), gp_dmd.ll_values)
    plt.show()


if __name__ == '__main__':
    demo()