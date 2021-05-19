import matplotlib.pyplot as plt
import argparse

from datasets import MinimumJerk
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
    This demo shows an example of GP-DMD on the minimum_jerk_trajectories data.
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

    y_init = jnp.deg2rad(jnp.array([[-10.], [50.]]))
    y_final = jnp.deg2rad(jnp.array([[50.], [-20.]]))
    data = MinimumJerk(y_init, y_final, t_steps, s_size=1, sigma=args.sigma, prng_handler=prng_handler)
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
    ax12.set_title(f"Target Data")
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

    # Figure
    _samples = gp_dmd.get_sample(labels_x, labels_idx, t_steps, 100)
    fig2 = plt.figure()
    grid2 = fig2.add_gridspec(3, 2)
    ax200 = fig2.add_subplot(grid2[0, 0])
    ax200.set_title(f"Position 1. Joint")
    ax200.set_xlabel("Time")
    ax200.set_ylabel("$x$")
    for s in _samples:
        ax200.plot(t_steps, jnp.real(s[0]), 'b', alpha=0.05)
    ax200.plot(t_steps, mu[:, 0], 'b', label='Mean')
    ax200.plot(t_steps, y[:, 0], '--r', label='Data')
    ax200.legend()

    ax201 = fig2.add_subplot(grid2[0, 1])
    ax201.set_title(f"Position 2. Joint")
    ax201.set_xlabel("Time")
    ax201.set_ylabel("$x$")
    for s in _samples:
        ax201.plot(t_steps, jnp.real(s[1]), 'b', alpha=0.05)
    ax201.plot(t_steps, mu[:, 1], 'b', label='Mean')
    ax201.plot(t_steps, y[:, 1], '--r', label='Data')
    ax201.legend()

    ax210 = fig2.add_subplot(grid2[1, 0])
    ax210.set_title(f"Velocity 1. Joint")
    ax210.set_xlabel("Time")
    ax210.set_ylabel("$v$")
    plt.gca().set_prop_cycle(None)
    for s in _samples:
        ax210.plot(t_steps, jnp.real(s[2]), 'b', alpha=0.05)
    ax210.plot(t_steps, mu[:, 2], 'b', label='Mean')
    ax210.plot(t_steps, y[:, 2], '--r', label='Data')
    ax210.legend()

    ax211 = fig2.add_subplot(grid2[1, 1])
    ax211.set_title(f"Velocity 2. Joint")
    ax211.set_xlabel("Time")
    ax211.set_ylabel("$v$")
    for s in _samples:
        ax211.plot(t_steps, jnp.real(s[3]), 'b', alpha=0.05)
    ax211.plot(t_steps, mu[:, 3], 'b', label='Mean')
    ax211.plot(t_steps, y[:, 3], '--r', label='Data')
    ax211.legend()

    ax220 = fig2.add_subplot(grid2[2, 0])
    ax220.set_title(f"Acceleration 1. Joint")
    ax220.set_xlabel("Time")
    ax220.set_ylabel("$a$")
    for s in _samples:
        ax220.plot(t_steps, jnp.real(s[4]), 'b', alpha=0.05)
    ax220.plot(t_steps, mu[:, 4], 'b', label='Mean')
    ax220.plot(t_steps, y[:, 4], '--r', label='Data')
    ax220.legend()

    ax221 = fig2.add_subplot(grid2[2, 1])
    ax221.set_title(f"Acceleration 2. Joint")
    ax221.set_xlabel("Time")
    ax221.set_ylabel("$a$")
    for s in _samples:
        ax221.plot(t_steps, jnp.real(s[5]), 'b', alpha=0.05)
    ax221.plot(t_steps, mu[:, 5], 'b', label='Mean')
    ax221.plot(t_steps, y[:, 5], '--r', label='Data')
    ax221.legend()
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