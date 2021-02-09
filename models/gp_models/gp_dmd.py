""" This scripts combines Gaussian Process Dynamical Models with Dynamic Mode Decomposition.

Reference:
Tu, Jonathan H., et al. "On dynamic mode decomposition: Theory and applications." arXiv preprint arXiv:1312.0041 (2013).
https://arxiv.org/abs/1312.0041

Wang, J. M., Fleet, D. J., & Hertzmann, A. (2005, December). Gaussian process dynamical models. In NIPS (Vol. 18, p. 3).
https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.85.2953&rep=rep1&type=pdf

Wang, J. M., Fleet, D. J., & Hertzmann, A. (2007). Gaussian process dynamical models for human motion.
IEEE transactions on pattern analysis and machine intelligence, 30(2), 283-298.
"""

import jax
import jax.numpy as jnp
import jax.random as random

from jax import jit
from utilities.utils import  pos_def
from utilities.optimizer_lib import Adam
from tqdm import tqdm
from numpyro.distributions import MultivariateNormal


class GPDMD(object):
    def __init__(self, **kwargs):
        """ Guassian Process Dynamic Mode Decomposition """
        self.sigma_x = None
        self.sigma_z = None
        self.a_tilde = None
        self.sigma_a = None

        self.mu = None
        self.phi = None

        self.b = None

        self.iterations = kwargs.get("iterations", None)

        self.latent_dim = kwargs.get("latent_dim", None)

        self.prng_handler = kwargs.get("prng_handler", None)

        self.optimizer = Adam(self.loss, **kwargs)

        self.kernel_fun = kwargs.get("kernel", None)

        self.ll_values = []

    def fit(self, x: jnp.ndarray, *args, **kwargs) -> None:
        """ Compute the Gaussian Process Dynamic Mode Decomposition given the data matrix
        :param x: Data matrix as jax numpy ndarray.
        """
        key = self.prng_handler.get_keys(1)
        variables = self._init_vars(key[0], x.T, self.latent_dim)

        _, self.sigma_x, z, self.sigma_z, a_tilde, self.sigma_a = self._train(variables)

        # Compute lower dimensional linear operator
        self.a_tilde = a_tilde.T

        # Compute DMD Values and DMD Modes
        self.mu, self.phi = jnp.linalg.eig(self.a_tilde)

        # Compute Amplitude of DMD Modes
        self.b = jnp.linalg.lstsq(self.phi, z.T[:, 0], rcond=None)[0]

    def predict(self, t_steps: jnp.ndarray, pow: float = 1) -> jnp.ndarray:
        """ Predict the data with the calculated DMD values
        :param t_steps: Time steps as jax.numpy.ndarray of size N
        :param pow: Exponential Power for vander matrix
        :return: Predicted Data as jax.numpy.ndarray
        """
        time_behaviour = self._vander(self.mu, len(t_steps), pow=pow)
        return self.phi @ jnp.diag(self.b) @ time_behaviour

    def _train(self, variables):
        x, sigma_x, z, sigma_z, a_tilde, sigma_a = variables
        opt_state = self.optimizer.opt_init([z, sigma_z])

        x_outer = jnp.einsum('kd, hd -> kh', x, x)
        for _ in tqdm(range(self.iterations)):
            args = [x_outer, sigma_x, a_tilde, sigma_a, [*x.shape, self.latent_dim]]
            params, opt_state, value = self.optimizer.step([z, sigma_z], opt_state, self.kernel_fun, *args)
            z, sigma_z = params
            sigma_z = jnp.maximum(sigma_z, 1e-2)

            args = [sigma_x, z]
            a_tilde, sigma_a = self._close_form_step([a_tilde, sigma_a], *args)
            sigma_a = jnp.maximum(sigma_a, 1e-2)

            self.ll_values.append(value)

        return [x, sigma_x, z, sigma_z, a_tilde, sigma_a]

    def get_mean_sigma(self, labels_x, labels_idx, t_steps):
        z = self.predict(t_steps).T

        K11 = self.kernel_fun.transform(z[labels_idx, :].T, z[labels_idx, :].T)
        K12 = self.kernel_fun.transform(z[labels_idx, :].T, z.T)
        K21 = K12.T
        K22 = self.kernel_fun.transform(z.T, z.T)

        cholesky_inv = jnp.linalg.inv(jnp.linalg.cholesky(K11 + jnp.eye(*K11.shape) * self.sigma_z))

        return K21 @ cholesky_inv.conj().T @ cholesky_inv @ labels_x, K22 - K21 @ cholesky_inv.conj().T @ cholesky_inv @ K12

    def get_sample(self, labels_x, labels_idx, t_steps, number):

        mean, sigma = self.get_mean_sigma(labels_x, labels_idx, t_steps)

        mvnrml = MultivariateNormal(loc=mean.T, covariance_matrix=pos_def(sigma))
        return mvnrml.sample(self.prng_handler.get_keys(1)[0], sample_shape=(number,))

    @staticmethod
    def _init_vars(keys, x, latent_dim):
        """  This method initializes a sample given random keys, observations and the latent_dim

        :param keys (Jax device array): Pseudo random number generator (PRNG) Keys
        :param x (Jax devicearray): Contains a jax devicearray corresponding to the data
        :param latent_dim (int): Dimensionality of the latent space.
        :return:
        """
        m, _ = x.shape
        sigma_x = 1

        z = jnp.ones(shape=(m, latent_dim))
        sigma_z = 1e-2

        a_tilde = random.normal(key=keys, shape=(latent_dim, latent_dim))
        sigma_a = 1e-2
        return [x, sigma_x, z, sigma_z, a_tilde, sigma_a]

    @staticmethod
    @jit
    def _close_form_step(params, *args):
        a_tilde, sigma_a = params
        sigma_x, z = args

        latent_dim = a_tilde.shape[0]

        a_tilde = jnp.linalg.pinv(sigma_x / sigma_a + z[:-1, :].T @ z[:-1, :]) @ z[:-1, :].T @ z[1:, :]

        sigma_a = 1 / (latent_dim ** 2 - 1) * jnp.trace(a_tilde @ a_tilde.T)
        return a_tilde, sigma_a

    @staticmethod
    @jax.partial(jit, static_argnums=(1,))
    def loss(params, kernel_fun, *args):
        z, sigma_z = params
        x_outer, sigma_x, a_tilde, sigma_a, dims = args

        m, n, latent_dim = dims

        kernel = kernel_fun.transform(z.T, z.T)
        kernel += jnp.eye(*kernel.shape) * sigma_z
        chol = jnp.linalg.cholesky(kernel)
        cholinv = jnp.linalg.inv(chol)
        kernelinv = cholinv.T @ cholinv

        loss = 1 / 2 * (m * latent_dim * jnp.log(sigma_x) + latent_dim ** 2 * jnp.log(sigma_a)
                        + (1 / sigma_a) * jnp.trace(a_tilde @ a_tilde.T)
                        + n * jnp.sum(jnp.log(jnp.linalg.det(kernel))) + jnp.einsum('nn -> ', kernelinv @ x_outer)
                        + (1 / sigma_x) * jnp.trace(z.T @ z - 2 * z[:-1, :].T @ z[1:, :] @ a_tilde.T
                                                    + z[:-1, :].T @ z[:-1, :] @ a_tilde @ a_tilde.T))
        return loss

    @staticmethod
    def _vander(x: jnp.ndarray, n: int, pow: float=1) -> jnp.ndarray:
        """ Generate a Vandermonde matrix.
        :param x: jax.numpy.ndarray
        :param n: int describes the number of columns
        :param pow: float corresponding to the power of x per step
        :return: a Vandermond matrix  as jax.numpy.ndarray
        """
        return jnp.column_stack([x ** (i * pow) for i in range(n)])