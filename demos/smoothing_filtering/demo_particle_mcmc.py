import matplotlib.pyplot as plt
import argparse

from utilities.prng_handler import PRNGHandler
from numpyro.distributions import MultivariateNormal
from utilities.inference_lib import *


parser = argparse.ArgumentParser(description="Test")
parser.add_argument("--device", default="cpu", type=str,
                    help="Specify the device on which Jax works. Possible devices are cpu or gpu (if available).")
parser.add_argument("--x64", default=True, type=bool,
                    help="Defines the float type used by Jax.")

parser.add_argument("--seed", default=11, type=int,
                    help="Defines the seed of the pseudo random number generator.")


def demo():
    """
    This demo shows an example of particle mcmc on two simplex examples
    :return: None
    """
    args = parser.parse_args()

    jax.config.update('jax_platform_name', args.device)
    jax.config.update('jax_enable_x64', args.x64)

    prng_handler = PRNGHandler(seed=args.seed)

    # ----------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------ Example 1 -----------------------------------------------------
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

    def pi_x0(key, mu, sigma, samples):
        nrml = MultivariateNormal(loc=mu, covariance_matrix=sigma)
        return nrml.sample(key, sample_shape=(samples, ))

    @jit
    def p_y(mu, sigma, y):
        nrml = MultivariateNormal(loc=func2(mu).T, covariance_matrix=sigma)
        return jnp.exp(nrml.log_prob(y))

    @jit
    def pi_x(key, mu, sigma):
        nrml = MultivariateNormal(loc=func(mu).T, covariance_matrix=sigma)
        return nrml.sample(key)

    @jit
    def p_x(mu, sigma, y):
        nrml = MultivariateNormal(loc=func(mu).T, covariance_matrix=sigma)
        return jnp.exp(nrml.log_prob(y))

    @jax.partial(jit, static_argnums=(2,))
    def cat(key, weights, samples):
        categorical = Categorical(weights)
        return categorical.sample(key, sample_shape=(samples,))

    @jax.partial(jit, static_argnums=(0,))
    def _weighting(func, mu, sigma, y, w_old=1):
        weights = w_old * func(mu, sigma, y)
        weights /= jnp.sum(weights)
        return weights

    num_particles = 20
    n, _ = X.shape

    initial = jnp.array([0, 1]).reshape(1, -1)
    X_hat = jnp.ones_like(X) * jnp.array([0, 1])[:, None]
    sigma_x = jnp.eye(n) * 1e-4
    sigma_y = jnp.eye(1) * 0.025

    pmcmc = ParticleMCMC(pi_x0, pi_x, p_x, p_y, initial[0], (sigma_x, sigma_y),
                       num_particles=num_particles, iterations=len(t_avail), prng_handler=prng_handler)
    pmcmc.filtering(_Y, X_hat)

    particles = pmcmc.particles
    X_hat = pmcmc.x_reference

    # ----------------------------------------------- Visualization ---------------------------------------------------

    fig = plt.figure()
    grid = fig.add_gridspec(2, 2)
    ax10 = fig.add_subplot(grid[0, :])
    ax10.set_title(f"Initial Samples")
    ax10.set_xlabel("Time $t$")
    ax10.set_ylabel("$f(t)$")
    ax10.plot(t_avail, _Y.T, 'go', alpha=0.25, label="Data")
    ax10.plot(t_avail, X[0], color=[0, 0, 0], alpha=0.25, label="Goal")

    test = jnp.array(particles)
    for i in range(test.shape[1]):
        ax10.plot(t_avail, test[:, i, 0], '--r', alpha=0.25)

    ax10.plot(t_avail, X_hat[0], 'b', label="Reference")
    plt.legend()

    # ----------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------ Example 2 -----------------------------------------------------
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

    def pi_x0(key, mu, sigma, samples):
        nrml = MultivariateNormal(loc=mu, covariance_matrix=sigma)
        return nrml.sample(key, sample_shape=(samples, ))

    @jit
    def p_y(mu, sigma, y):
        nrml = MultivariateNormal(loc=func2(mu).T, covariance_matrix=sigma)
        return jnp.exp(nrml.log_prob(y))

    @jit
    def pi_x(key, mu, sigma):
        nrml = MultivariateNormal(loc=func(mu).T, covariance_matrix=sigma)
        return nrml.sample(key)

    @jit
    def p_x(mu, sigma, y):
        nrml = MultivariateNormal(loc=func(mu).T, covariance_matrix=sigma)
        return jnp.exp(nrml.log_prob(y))

    @jax.partial(jit, static_argnums=(2,))
    def cat(key, weights, samples):
        categorical = Categorical(weights)
        return categorical.sample(key, sample_shape=(samples,))

    @jax.partial(jit, static_argnums=(0,))
    def _weighting(func, mu, sigma, y, w_old=1):
        weights = w_old * func(mu, sigma, y)
        weights /= jnp.sum(weights)
        return weights

    num_particles = 20
    n, _ = X.shape

    initial = jnp.array([0, 1]).reshape(1, -1)
    X_hat = jnp.ones_like(X) * jnp.array([0, 1])[:, None]
    sigma_x = jnp.eye(n) * 1e-4
    sigma_y = jnp.eye(1) * 0.025

    pmcmc = ParticleMCMC(pi_x0, pi_x, p_x, p_y, initial[0], (sigma_x, sigma_y),
                       num_particles=num_particles, iterations=len(t_avail), prng_handler=prng_handler)
    pmcmc.filtering(_Y, X_hat)

    particles = pmcmc.particles
    X_hat = pmcmc.x_reference

    # ----------------------------------------------- Visualization ---------------------------------------------------

    ax20 = fig.add_subplot(grid[1, :])
    ax20.set_title(f"Initial Samples")
    ax20.set_xlabel("Time $t$")
    ax20.set_ylabel("$sin(t)$")
    ax20.plot(t_avail, _Y.T, 'go', alpha=0.25, label="Data")
    ax20.plot(t_avail, X[0], color=[0, 0, 0], alpha=0.25, label="Goal")

    test = jnp.array(particles)
    for i in range(test.shape[1]):
        ax20.plot(t_avail, test[:, i, 0], '--r', alpha=0.25)

    ax20.plot(t_avail, X_hat[0], 'b', label="Reference")
    plt.legend()
    plt.show()


if __name__ == '__main__':
    demo()
