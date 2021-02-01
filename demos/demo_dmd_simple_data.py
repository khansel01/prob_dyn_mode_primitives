import matplotlib.pyplot as plt
import argparse

from jax.ops import index, index_update
from models.dmd_models import *
from utilities.utils import data2snapshots
from utilities.prng_handler import PRNGHandler
from utilities.kernel_lib import *


parser = argparse.ArgumentParser(description="Test")
parser.add_argument("--device", default="cpu", type=str,
                    help="Specify the device on which Jax works. Possible devices are cpu or gpu (if available).")
parser.add_argument("--x64", default=True, type=bool,
                    help="Defines the float type used by Jax.")

parser.add_argument("--dmd", default="Kernel", type=str,
                    help="Specifies the DMD method used. Possible Methods are "
                         "Standard, Exact, FB, TLS, Kernel, Probabilistic and Bayesian")

parser.add_argument("--kernel", default="RBF", type=str,
                    help="Specifies the Kernel used in KernelDMD. Possible Methods are Polynomial and RBF")

parser.add_argument("--trunc_svd", default=-1, type=int,
                    help="Specifies the truncation value for SVD in Standard-, Exact- and TLSDMD")
parser.add_argument("--trunc_tls", default=0, type=int,
                    help="Specifies the truncation value for TLS in Standard-, Exact- and FBDMD")

parser.add_argument("--seed", default=11, type=int,
                    help="Defines the seed of the pseudo random number generator.")
parser.add_argument("--samples", default=1, type=int,
                    help="Set the number of sampled trajectories.")
parser.add_argument("--time_delay", default=1, type=int,
                    help="Set an integer to define the time shift used to create delayed coordinates.")
parser.add_argument("--axis", default=1, type=int,
                    help="Set the axis along which the trajectories will be concatenated.")

parser.add_argument("--gibbs_iter", default=5000, type=int,
                    help="Specifies the iterations of the Gibbs sampler.")
parser.add_argument("--gibbs_burn_in", default=1000, type=int,
                    help="Specifies the Burn-in phase of the Gibbs sampler.")

parser.add_argument("--latent_dim", default=2, type=int,
                    help="Dimensionality of the latent space.")
parser.add_argument("--alpha", default=1e-3, type=float,
                    help="Initial shape of the gamma distribution.")
parser.add_argument("--beta", default=1e-3, type=float,
                    help="Initial rate of the gamma distribution.")


def demo():
    """
    This demo shows an evaluation of a specific DMD method on a simple data set.
    :return: None
    """
    args = parser.parse_args()

    jax.config.update('jax_platform_name', args.device)
    jax.config.update('jax_enable_x64', args.x64)

    t_steps = jnp.arange(0, 1, 0.01)

    def data():
        m = 100
        A = jnp.array([[1., 1.], [-1., 2.]])
        A /= jnp.sqrt(3)
        n = 2
        X = jnp.zeros((n, m))
        X = index_update(X, index[:, 0], jnp.array([-1, 1.]))
        for k in range(1, m):
            X = index_update(X, index[:, k], jnp.dot(A, X[:, k-1]))
        return X
    data = data()[None, :]
    x0, x1 = data2snapshots(data, t_delay=args.time_delay, axis=args.axis)

    kernel_type = eval(f'{args.kernel}Kernel')

    prng_handler = PRNGHandler(seed=args.seed)

    dmd_type = eval(f'{args.dmd}DMD')
    dmd = dmd_type(kernel=kernel_type(),
                   gibbs_iter=args.gibbs_iter, gibbs_burn_in=args.gibbs_burn_in,
                   prng_handler=prng_handler, latent_dim=args.latent_dim,
                   alpha=args.alpha, beta=args.beta)

    dmd.fit(x0, x1, trunc_svd=args.trunc_svd, trunc_tls=args.trunc_tls)
    out = jnp.real(dmd.predict(t_steps))

    # Visualize Data
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.set_title(f"Reconstructed Data")
    ax.set_xlabel("Time: $t$")
    ax.set_ylabel("Data: $y$")

    ax.plot(t_steps, out.T, '--', label="Predicted_Data", linewidth=2.0)
    plt.gca().set_prop_cycle(None)
    ax.plot(t_steps, data[0].T, alpha=0.3, label="Data", linewidth=2.5)
    ax.legend()

    plt.suptitle(f"{args.dmd}DMD evaluated on a simple Dataset")
    plt.show()


if __name__ == '__main__':
    demo()
