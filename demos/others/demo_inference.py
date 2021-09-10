import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

from numpyro.distributions import MultivariateNormal
from utils.utils_inference import filtering, smoothing


def demo() -> None:
    """
    This demo shows an example of the combination of Kalman filtering, filtering based on spherical cubature
    integration and subsequently Rauch—Tung—Striebel (RTS) smoothing on two simple data.
    :return: None
    """

    jax.config.update('jax_platform_name', 'cpu')
    jax.config.update('jax_enable_x64', True)

    t_steps = jnp.arange(0, 2 * jnp.pi, 0.05)
    x_init = jnp.array([0., 1., 0., 1.])[:, None]
    x0_sigma, fx_sigma, gx_sigma = jnp.eye(4), jnp.eye(4), jnp.eye(2)*1000

    # ------------------------- initialize linear operator, transition model and latent space -------------------------

    delta_t = t_steps[1] - t_steps[0]
    A = jnp.array([[1, delta_t, 0, 0], [0, 1, 0, 0], [0, 0, 1, delta_t], [0, 0, -delta_t, 1]])

    def fx(_x: jnp.ndarray) -> jnp.ndarray:
        return A @ _x

    x = jax.lax.scan(lambda init, xs: (jax.jit(fx)(init), init), x_init, xs=t_steps[:, None])[1][:, :, 0].T

    # --------------------------------- initialize emission model and noisy observations ------------------------------

    def gx(_x: jnp.ndarray) -> jnp.ndarray:
        return jnp.array([[1., 0., 0., 0.], [0., 0., 1., 0]]) @ _x

    mvnrml = MultivariateNormal(loc=jnp.zeros(1), covariance_matrix=jnp.eye(1) * 0.025)
    data = jax.jit(gx)(x) + mvnrml.sample(jax.random.PRNGKey(11), sample_shape=(len(t_steps), ))[:, 0]

    # ---------------------------------------- apply filtering and smoothing ------------------------------------------

    filter_results = jax.jit(filtering, static_argnums=(5, ), device=jax.devices()[0])(data=data.T,
                                                                                       x_inital=x_init,
                                                                                       inital_sigma=x0_sigma, fx=A,
                                                                                       fx_sigma=fx_sigma, gx=gx,
                                                                                       gx_sigma=gx_sigma)
    smoother_results = jax.jit(smoothing, device=jax.devices()[0])(f_means=filter_results[0],
                                                                   f_sigmas=filter_results[1], fx=A, fx_sigma=fx_sigma)

    filtered_mean = filter_results[0][:, :, 0]
    smoothed_mean = smoother_results[0][:, :, 0]

    # ----------------------------------------------- Visualization ---------------------------------------------------

    fig = plt.figure()

    plt.suptitle(f"Cubature Smoothing")

    grid = fig.add_gridspec(2, 2)

    ax10 = fig.add_subplot(grid[0, :])

    ax10.set_title(f"Initial Samples")
    ax10.set_xlabel("Time $t$")
    ax10.set_ylabel("$f(t)$")

    ax10.plot(t_steps, data[0], 'go', alpha=0.25, label="Data")
    ax10.plot(t_steps, x[0], color=[0, 0, 0], alpha=0.25, label="Goal")
    ax10.plot(t_steps, filtered_mean[:, 0], 'r', alpha=0.5, label="Filtered")
    ax10.plot(t_steps, smoothed_mean[:, 0], 'b', label="Smoothed")

    plt.legend()

    ax20 = fig.add_subplot(grid[1, :])

    ax20.set_title(f"Initial Samples")
    ax20.set_xlabel("Time $t$")
    ax20.set_ylabel("$sin(t)$")

    ax20.plot(t_steps, data[1], 'go', alpha=0.25, label="Data")
    ax20.plot(t_steps, x[2], color=[0, 0, 0], alpha=0.25, label="Goal")
    ax20.plot(t_steps, filtered_mean[:, 2], 'r', alpha=0.5, label="Filtered")
    ax20.plot(t_steps, smoothed_mean[:, 2], 'b', label="Smoothed")

    plt.legend()

    plt.show()

    return None


if __name__ == '__main__':
    demo()

