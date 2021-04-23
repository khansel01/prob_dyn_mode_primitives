import matplotlib.pyplot as plt
import argparse

from utilities.prng_handler import PRNGHandler
from utilities.kernel_lib import *
from numpyro.distributions import MultivariateNormal
from jax.ops import index_update, index
from utilities.inference_lib import Curbature


parser = argparse.ArgumentParser(description="Test")
parser.add_argument("--device", default="cpu", type=str,
                    help="Specify the device on which Jax works. Possible devices are cpu or gpu (if available).")
parser.add_argument("--x64", default=True, type=bool,
                    help="Defines the float type used by Jax.")
parser.add_argument("--seed", default=11, type=int,
                    help="Defines the seed of the pseudo random number generator.")


def demo():
    """
    This demo shows an example of curbature smoothing on two simplex examples
    :return: None
    """
    args = parser.parse_args()

    jax.config.update('jax_platform_name', args.device)
    jax.config.update('jax_enable_x64', args.x64)

    prng_handler = PRNGHandler(seed=args.seed)

    # ----------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------ Example 1 -----------------------------------------------------
    # ----------------------------------------------------------------------------------------------------------------

    mean = jnp.array([80., 0.8])[:, None]
    sigma = jnp.diag(jnp.array([40., 0.4]))
    mvnrml = MultivariateNormal(loc=mean.T, covariance_matrix=sigma)
    samples_init = mvnrml.sample(prng_handler.get_keys(1)[0], sample_shape=(500,))[:, 0, :].T

    def transform(_s):
        x = _s[0] * jnp.cos(_s[1])
        y = _s[0] * jnp.sin(_s[1])
        return jnp.append(x[None, :], y[None, :], axis=0)

    samples_transformed = transform(samples_init)
    n, m = mean.shape
    xi = jnp.zeros((n, 2*n))
    xi = index_update(xi, index[:, :n], jnp.eye(n) * jnp.sqrt(n))
    xi = index_update(xi, index[:, n:], - jnp.eye(n) * jnp.sqrt(n))
    sigma_points_before = mean + jnp.linalg.cholesky(sigma)@xi

    sigma_points_after = transform(sigma_points_before)

    # ----------------------------------------------- Visualization ---------------------------------------------------

    fig = plt.figure()
    plt.suptitle(f"Curbature Smoothing")
    grid = fig.add_gridspec(3, 2)
    ax00 = fig.add_subplot(grid[0, 0])
    ax00.set_title(f"Initial Samples")
    ax00.set_xlabel("$r$")
    ax00.set_ylabel("$\Theta$")
    ax00.plot(samples_init[0], samples_init[1], 'go', alpha=0.25, label="Samples")
    ax00.plot(sigma_points_before[0], sigma_points_before[1], 'rx', markersize=10., lw=10., label="Sigma Points")
    plt.legend()

    ax01 = fig.add_subplot(grid[0, 1])
    ax01.set_title(f"Transformed Sample")
    ax01.set_xlabel("$x$")
    ax01.set_ylabel("$y$")
    ax01.plot(samples_transformed[0], samples_transformed[1], 'go', alpha=0.25, label="Samples")
    ax01.plot(sigma_points_after[0], sigma_points_after[1], 'rx', markersize=10., lw=10., label="Sigma Points")
    plt.legend()

    # ----------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------ Example 2 -----------------------------------------------------
    # ----------------------------------------------------------------------------------------------------------------

    t_steps = jnp.arange(0, 2 * jnp.pi, 0.01)
    t_avail = t_steps[::5]
    delta_t = t_avail[1] -t_avail[0]
    mvnrml = MultivariateNormal(loc=jnp.zeros(1), covariance_matrix=jnp.eye(1)*0.025)
    C = jnp.array([1, 0])[None, :]
    A = jnp.array([1, delta_t, 0, 1]).reshape(2, 2)

    @jit
    def func(_x):
        return A @ _x

    @jit
    def func2(_x):
        return C @ _x

    X = [jnp.array([0, 1])]
    for i in range(1, len(t_avail)):
        X.append(func(X[-1]))
    X = jnp.array(X).T
    _Y = C @ X + mvnrml.sample(prng_handler.get_keys(1)[0], sample_shape=(len(t_avail), ))[:, 0]

    x_init = jnp.array([0, 1])
    q = jnp.eye(2)
    r = jnp.eye(1)*10
    smoother = Curbature(fx=func, gx=func2, dims=(2, len(t_avail), 1), q=q, r=r, x_init=x_init)
    smoother.smoothing(labels=_Y)

    # ----------------------------------------------- Visualization ---------------------------------------------------

    ax10 = fig.add_subplot(grid[1, :])
    ax10.set_title(f"Initial Samples")
    ax10.set_xlabel("Time $t$")
    ax10.set_ylabel("$f(t)$")
    ax10.plot(t_avail, _Y.T, 'go', alpha=0.25, label="Data")
    ax10.plot(t_avail, X[0], color=[0, 0, 0], alpha=0.25, label="Goal")
    ax10.plot(t_avail, smoother.filtered_mean[0], 'r', alpha=0.5, label="Filtered")
    ax10.plot(t_avail, smoother.smoothed_mean[0], 'b', label="Smoothed")
    plt.legend()

    # ----------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------ Example 3 -----------------------------------------------------
    # ----------------------------------------------------------------------------------------------------------------

    t_steps = jnp.arange(0, 2 * jnp.pi, 0.01)
    t_avail = t_steps[::5]
    delta_t = t_avail[1] -t_avail[0]
    mean = jnp.zeros(1)
    sigma = jnp.eye(1)*0.025
    mvnrml = MultivariateNormal(loc=mean, covariance_matrix=sigma)
    C = jnp.array([1, 0])[None, :]
    A = jnp.array([1, delta_t, -delta_t, 1]).reshape(2, 2)

    @jit
    def func(_x):
        return A @ _x

    @jit
    def func2(_x):
        return C @ _x

    X = [jnp.array([0, 1])]
    for i in range(1, len(t_avail)):
        X.append(func(X[-1]))
    X = jnp.array(X).T
    _Y = C @ X + mvnrml.sample(prng_handler.get_keys(1)[0], sample_shape=(len(t_avail), ))[:, 0]

    # CKF
    # Init
    x_init = jnp.array([0, 1])
    q = jnp.eye(2)
    r = jnp.eye(1)*10
    smoother = Curbature(fx=func, gx=func2, dims=(2, len(t_avail), 1), q=q, r=r, x_init=x_init)
    smoother.smoothing(labels=_Y)

    # ----------------------------------------------- Visualization ---------------------------------------------------

    ax20 = fig.add_subplot(grid[2, :])
    ax20.set_title(f"Initial Samples")
    ax20.set_xlabel("Time $t$")
    ax20.set_ylabel("$sin(t)$")
    ax20.plot(t_avail, _Y.T, 'go', alpha=0.25, label="Data")
    ax20.plot(t_avail, X[0], color=[0, 0, 0], alpha=0.25, label="Goal")
    ax20.plot(t_avail, smoother.filtered_mean[0], 'r', alpha=0.5, label="Filtered")
    ax20.plot(t_avail, smoother.smoothed_mean[0], 'b', label="Smoothed")
    plt.legend()
    plt.show()


if __name__ == '__main__':
    demo()