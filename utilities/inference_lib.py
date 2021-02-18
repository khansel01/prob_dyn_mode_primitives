from utilities.utils import pos_def
from utilities.kernel_lib import *
from jax.ops import index_update, index
from numpyro.distributions import Categorical


class Curbature(object):
    def __init__(self, fx: jnp.function, gx: jnp.function, dims: tuple, **kwargs):
        self.fx = fx
        self.gx = gx

        self.dims = dims

        self.q = kwargs.get('q', jnp.eye(self.dims[0]))
        self.r = kwargs.get('r', jnp.eye(self.dims[2]))

        self.xi = jnp.zeros((self.dims[0], 2 * self.dims[0]))
        self.xi = index_update(self.xi, index[:, :self.dims[0]], jnp.eye(self.dims[0]) * jnp.sqrt(self.dims[0]))
        self.xi = index_update(self.xi, index[:, self.dims[0]:], - jnp.eye(self.dims[0]) * jnp.sqrt(self.dims[0]))

        self.x_init = kwargs.get('x_init', jnp.zeros(self.dims[0]))

        self.filtered_mean = jnp.zeros((self.dims[0], self.dims[1]))
        self.filtered_mean = index_update(self.filtered_mean, index[:, 0], self.x_init)
        self.filtered_sigma = jnp.zeros((self.dims[1], self.dims[0], self.dims[0]))
        self.filtered_sigma = index_update(self.filtered_sigma, index[0], jnp.eye(2))

        self.smoothed_mean = None
        self.smoothed_sigma = None

    def filtering(self, labels):
        args = [self.xi, self.q, self.r]
        for step in range(1, self.dims[1]):
            mu, sigma = self._forward(self.filtered_mean[:, step - 1, None], self.filtered_sigma[step - 1],
                                      labels[:, step], self.fx, self.gx, *args)

            self.filtered_mean = index_update(self.filtered_mean, index[:, step:step + 1], mu)
            self.filtered_sigma = index_update(self.filtered_sigma, index[step], sigma)

        self.smoothed_mean = self.filtered_mean.copy()
        self.smoothed_sigma = self.filtered_sigma.copy()

    def smoothing(self, labels):
        self.filtering(labels)

        for step in range(self.dims[1] - 1, 0, -1):
            args = [self.smoothed_mean[:, step, None], self.smoothed_sigma[step], self.xi, self.q]

            mu, sigma = self._backward(self.smoothed_mean[:, step - 1, None], self.smoothed_sigma[step - 1],
                                       self.fx, *args)

            self.smoothed_mean = index_update(self.smoothed_mean, index[:, step - 1:step], mu)
            self.smoothed_sigma = index_update(self.smoothed_sigma, index[step - 1], sigma)

    @staticmethod
    @jax.partial(jit, static_argnums=(3, 4,))
    def _forward(mu: jnp.ndarray, sigma: jnp.ndarray, label, fx, gx, *args):
        xi, q, r = args
        n, _ = mu.shape

        # Prediction step
        s_p_x = fx(mu + jnp.linalg.cholesky(pos_def(sigma)) @ xi)
        mu_predicted = jnp.sum(s_p_x, axis=1)[:, None] / (2 * n)
        sigma_predicted = (s_p_x - mu_predicted) @ (s_p_x - mu_predicted).T / (2 * n) + q

        # Update step
        s_p_x = mu_predicted + jnp.linalg.cholesky(pos_def(sigma_predicted)) @ xi
        s_p_z = gx(s_p_x)
        mu_upd = jnp.sum(s_p_z, axis=1)[:, None] / (2 * n)
        sigma_upd = (s_p_z - mu_upd) @ (s_p_z - mu_upd).T / (2 * n) + r
        corr = (s_p_x - mu_predicted) @ (s_p_z - mu_upd).T / (2 * n)
        gain = corr @ jnp.linalg.inv(sigma_upd)

        return mu_predicted + gain @ (label - mu_upd), sigma_predicted + gain @ jnp.linalg.inv(sigma_upd) @ gain.T

    @staticmethod
    @jax.partial(jit, static_argnums=(2,))
    def _backward(mu, sigma, fx, *args):
        mu_smoothed, sigma_smoothed, xi, q = args
        n = mu.shape[0]

        s_p_b = mu + jnp.linalg.cholesky(pos_def(sigma)) @ xi
        s_p_a = fx(s_p_b)
        _mu = jnp.sum(s_p_a, axis=1)[:, None] / (2 * n)
        _cov = (s_p_a - _mu) @ (s_p_a - _mu).T / (2 * n) + q
        _cor = (s_p_b - mu) @ (s_p_a - _mu).T / (2 * n)
        gain = _cor @ jnp.linalg.inv(_cov)

        return mu + gain @ (mu_smoothed - _mu), sigma + gain @ jnp.linalg.inv(sigma_smoothed - _cov) @ gain.T


class ParticleSmoother(object):
    def __init__(self, pi_x0: jnp.function, pi_x: jnp.function, p_x: jnp.function
                 , p_y: jnp.function, x_init: jnp.array, sigmas: tuple, **kwargs):
        # sample distributions
        self.pi_x0 = pi_x0
        self.pi_x = pi_x

        # probability distributions
        self.p_x = p_x
        self.p_y = p_y

        # initial value
        self.x_init = x_init

        self.sigma_x, self.sigma_y = sigmas

        self.prng_handler = kwargs.get("prng_handler", None)

        self.num_particles = kwargs.get("num_particles", 20)
        self.iterations = kwargs.get("iterations", 100)

        self.particles = []
        self.weights = []
        self.smooth_particle = []

    def filtering(self, labels):
        self.particles.append(self.pi_x0(self.prng_handler.get_keys(1)[0], self.x_init,
                                         self.sigma_x, self.num_particles))

        self.weights.append(self._weighting(self.p_y, self.particles[-1].T,
                                            self.sigma_y, labels[:, 0]))

        for i in range(1, self.iterations):
            indx = self.cat(self.prng_handler.get_keys(1)[0], self.weights[-1], self.num_particles)

            self.particles.append(self.pi_x(self.prng_handler.get_keys(1)[0], self.particles[-1][indx].T,
                                            self.sigma_x))

            self.weights.append(self._weighting(self.p_y, self.particles[-1].T,
                                                self.sigma_y, labels[:, i]))

    def smoothing(self):
        indx = self.cat(self.prng_handler.get_keys(1)[0], self.weights[-1], 1)[0]
        self.smooth_particle.append(self.particles[-1][indx])

        for i in range(self.iterations - 1, 0, -1):
            weights = self._weighting(self.p_x, self.particles[i - 1].T, self.sigma_x,
                                           self.particles[i][indx], self.weights[i - 1])

            indx = self.cat(self.prng_handler.get_keys(1)[0], weights, 1)[0]
            self.smooth_particle.append(self.particles[i - 1][indx])

        self.smooth_particle.reverse()

    @staticmethod
    @jax.partial(jit, static_argnums=(2,))
    def cat(key, weights, samples):
        categorical = Categorical(weights)
        return categorical.sample(key, sample_shape=(samples,))

    @staticmethod
    @jax.partial(jit, static_argnums=(0,))
    def _weighting(func, mu, sigma, y, w_old=1):
        weights = w_old * func(mu, sigma, y)
        weights /= jnp.sum(weights)
        return weights


class PKSmoother(object):
    def __init__(self, fx, gx, p_y, x_init, sigmas, **kwargs):
        self.fx = fx
        self.gx = gx

        self.p_y = p_y

        self.x_init = x_init[:, None]
        self.sigma_x, self.sigma_y = sigmas

        self.q = kwargs.get('q', jnp.eye(self.x_init.shape[0]) * 1e-5)

        self.prng_handler = kwargs.get("prng_handler", None)
        self.iterations = kwargs.get("iterations", 100)

        self.filter_mu = []
        self.filter_sigma = []

        self.smooth_mu = []
        self.smooth_sigma = []

    def filtering(self, labels):
        sig_points = self.x_init + jnp.linalg.cholesky(pos_def(self.sigma_x)) @ self._xi

        weights = self._weighting(self.p_y, sig_points, self.sigma_y, labels[:, 0])

        self.filter_mu.append(weights @ sig_points.T)
        _delta = (sig_points.T - self.filter_mu[-1]).T
        self.filter_sigma.append(jnp.einsum(f'jn, in -> ji', weights * _delta, _delta))

        for i in range(1, self.iterations):
            _mu = self.fx @ self.filter_mu[-1].T
            _sigma = self.filter_sigma[-1] + self.sigma_x

            sig_points = _mu[:, None] + jnp.linalg.cholesky(pos_def(_sigma)) @ self._xi

            weights = self._weighting(self.p_y, sig_points, self.sigma_y, labels[:, i])
            self.filter_mu.append(weights @ sig_points.T)
            _delta = (sig_points.T - self.filter_mu[-1]).T
            self.filter_sigma.append(jnp.einsum(f'jn, in -> ji', weights * _delta, _delta))

    def smoothing(self, labels):
        self.filtering(labels)

        self.smooth_mu = self.filter_mu.copy()
        self.smooth_sigma = self.filter_sigma.copy()

        for i in range(self.iterations - 1, 0, -1):
            _mu, _sigma =  self._backward(self.fx,
                                          [self.smooth_mu[i - 1], self.smooth_mu[i]],
                                          [self.smooth_sigma[i - 1], self.smooth_sigma[i]],
                                          self.q)

            self.smooth_mu[i - 1], self.smooth_sigma[i - 1] = _mu, _sigma

    @property
    def _xi(self):
        xi = jnp.zeros((self.x_init.shape[0], 2 * self.x_init.shape[0]))
        xi = index_update(xi, index[:, :self.x_init.shape[0]],
                          jnp.eye(self.x_init.shape[0]) * jnp.sqrt(self.x_init.shape[0]))
        xi = index_update(xi, index[:, self.x_init.shape[0]:],
                          - jnp.eye(self.x_init.shape[0]) * jnp.sqrt(self.x_init.shape[0]))
        return xi

    @staticmethod
    @jit
    def _backward(fx, mu: list, sigma: list, *args):
        _mu = fx @  mu[0]
        _sigma = fx @ sigma[0] @ fx.T + args[0]
        _gain = sigma[0] @ fx.T @ jnp.linalg.inv(_sigma)

        return mu[0] + _gain @ (mu[1] - _mu), sigma[0] + _gain @ (sigma[1] - _sigma) @ _gain.T

    @staticmethod
    @jax.partial(jit, static_argnums=(0,))
    def _weighting(func, mu, sigma, y, w_old=1):
        weights = w_old * func(mu, sigma, y)
        weights /= jnp.sum(weights)
        return weights