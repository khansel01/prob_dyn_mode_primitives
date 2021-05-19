import matplotlib.pyplot as plt
import argparse

from datasets import EightShapes
from utilities.utils import Whitening
from utilities.prng_handler import PRNGHandler
from utilities.kernel_lib import *
from models.gp_models.gp_dmd import GPDMD

parser = argparse.ArgumentParser(description="Test")
parser.add_argument("--device", default="cpu", type=str,
                    help="Specify the device on which Jax works. Possible devices are cpu or gpu (if available).")
parser.add_argument("--x64", default=True, type=bool,
                    help="Defines the float type used by Jax.")

parser.add_argument("--seed", default=11, type=int,
                    help="Defines the seed of the pseudo random number generator.")
parser.add_argument("--sigma", default=0.1, type=float,
                    help="Set the variability in the sampled data.")

parser.add_argument("--latent_dim", default=5, type=int,
                    help="Dimensionality of the latent space.")
parser.add_argument("--iterations", default=50000, type=int,
                    help="Specifies the iterations of the Optimizer.")
parser.add_argument("--l_r", default=1e-3, type=float,
                    help="Specifies the learning rate of the Optimizer.")


def demo():
    """
    This demo shows an example of GP-DMD on the eight shape data set.
    :return: None
    """
    args = parser.parse_args()

    jax.config.update('jax_platform_name', args.device)
    jax.config.update('jax_enable_x64', args.x64)

    t_steps = jnp.arange(0, 1, 0.01)

    prng_handler = PRNGHandler(seed=args.seed)

    # -----------------------------------------------------------------------------------------------------------------
    # ---------------------------------------------Generate Data-------------------------------------------------------
    # -----------------------------------------------------------------------------------------------------------------

    data = EightShapes(t_steps, s_size=1, prng_handler=prng_handler)
    y = jnp.concatenate(data.transform, axis=1).T

    standarizer = Whitening(y.reshape(1, -1, order='F'))
    y = jnp.float64(standarizer().T.reshape(*y.shape, order='F'))

    y = y.reshape(data.transform.transpose((0, 2, 1)).shape)
    y = y[0]

    # -----------------------------------------------------------------------------------------------------------------
    # --------------------------------------------------Init-----------------------------------------------------------
    # -----------------------------------------------------------------------------------------------------------------

    alpha_x = 1.
    beta_x = 1e-3
    alpha_0 = 1.
    beta_0 = 1e-3
    alpha_a = 1.
    beta_a = 1e-3
    alpha_y = 1.
    beta_y = 1e-3

    @jit
    def kernel_fun(_param, _x, _y):
        i = _x.shape[1]
        j = _y.shape[1]
        inner = - 2 * jnp.einsum('ni, nj -> ij', _x, _y) + \
            jnp.ones((i, j)) * jnp.diag(jnp.einsum('ni, nj -> ij', _y, _y)) + \
            (jnp.ones((j, i)) * jnp.diag(jnp.einsum('ni, nj -> ij', _x, _x))).T
        return _param[1] * jnp.exp(-_param[0]/2 * inner)

    gp_dmd = GPDMD(iterations=args.iterations, latent_dim=args.latent_dim, l_r=args.l_r,
                   prng_handler=prng_handler, kernel=kernel_fun, alpha_0=alpha_0, beta_0=beta_0, alpha_x=alpha_x,
                   beta_x=beta_x, alpha_y=alpha_y, beta_y=beta_y, alpha_a=alpha_a, beta_a=beta_a)

    # -----------------------------------------------------------------------------------------------------------------
    # ---------------------------------------------- Training ---------------------------------------------------------
    # -----------------------------------------------------------------------------------------------------------------

    gp_dmd.fit(y)

    # -----------------------------------------------------------------------------------------------------------------
    # ---------------------------------- Prediction Observation & Latent Space ----------------------------------------
    # -----------------------------------------------------------------------------------------------------------------

    labels_idx = [0, 33, 66, 99]
    labels_x = y[labels_idx, :]

    x_latent = gp_dmd.predict(t_steps)
    _samples = gp_dmd.get_sample(labels_x, labels_idx, t_steps, 10)
    mu, _ = gp_dmd.get_mean_sigma(labels_x, labels_idx, t_steps)

    # -----------------------------------------------------------------------------------------------------------------
    # ----------------------------------------------- Visualization ---------------------------------------------------
    # -----------------------------------------------------------------------------------------------------------------

    fig = plt.figure()
    grid = fig.add_gridspec(5, 1)

    ax11 = fig.add_subplot(grid[0, 0])
    ax11.set_title(f"Target Data")
    ax11.set_xlabel("Time")
    ax11.set_ylabel("$\Theta$")
    plt.gca().set_prop_cycle(None)
    ax11.plot(t_steps, y)

    ax12 = fig.add_subplot(grid[1, 0])
    ax12.set_title(f"Predicted Mean")
    ax12.set_xlabel("Time")
    ax12.set_ylabel("$\Theta$")
    plt.gca().set_prop_cycle(None)
    ax12.plot(t_steps, mu)

    ax13 = fig.add_subplot(grid[2, 0])
    ax13.set_title(f"Samples")
    ax13.set_xlabel("Time")
    ax13.set_ylabel("$\Theta$")
    for s in _samples:
        plt.gca().set_prop_cycle(None)
        ax13.plot(t_steps, jnp.real(s.T))

    ax14 = fig.add_subplot(grid[3, 0])
    ax14.set_title(f"Latent_dim")
    ax14.set_xlabel("Time")
    ax14.set_ylabel("$\Theta$")
    ax14.plot(t_steps, jnp.real(x_latent), '--')

    ax15 = fig.add_subplot(grid[4, 0])
    ax15.set_title("Log_Likelihood")
    ax15.set_xlabel("Sample")
    ax15.set_ylabel("$log_likelihood value$")
    ax15.plot(jnp.arange(len(gp_dmd.ll_values)), gp_dmd.ll_values)

    fig2 = plt.figure()
    grid2 = fig2.add_gridspec(1, 1)
    ax2 = fig2.add_subplot(grid2[0, 0], projection='3d')
    _samples = gp_dmd.get_sample(labels_x, labels_idx, t_steps, 100)
    for i, s in enumerate(_samples):
        if i == 0:
            ax2.plot(s[0], s[1], s[2], color='blue', alpha=0.05)
        else:
            ax2.plot(s[0], s[1], s[2], color='blue', alpha=0.05)
    ax2.plot(mu[:, 0], mu[:, 1], mu[:, 2], label='Mean', color='blue')
    ax2.plot(y[:, 0], y[:, 1], y[:, 2], label='Data', color='red')
    ax2.set_title("GPDMD Eight-Shape Dataset")
    ax2.set_xlabel("$x$")
    ax2.set_ylabel("$y$")
    ax2.set_zlabel("$z$")
    ax2.legend()
    plt.show()

    mse_mean = jnp.trace((mu-y)@(mu-y).T)/mu.shape[0]
    mse_samples = []
    for s in _samples:
        mse_samples.append(jnp.trace((s.T - y)@(s.T - y).T) / mu.shape[0])
    delta_mse = jnp.array(mse_samples) - mse_mean
    mse_std = jnp.sqrt(jnp.sum(delta_mse**2)/len(mse_samples))
    print(f'The Mean Squared Error: {jnp.absolute(mse_mean)}')
    print(f'The Standard Deviation of the Mean Squared Error: {jnp.absolute(mse_std)}')


if __name__ == '__main__':
    demo()
