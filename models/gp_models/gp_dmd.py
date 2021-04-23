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
        self.alpha_x = kwargs.get("alpha_x", 1.)
        self.beta_x = kwargs.get("beta_x", 1e-3)
        self.alpha_0 = kwargs.get("alpha_0", 1.)
        self.beta_0 = kwargs.get("beta_0", 1e-3)
        self.alpha_a = kwargs.get("alpha_a", 1.)
        self.beta_a = kwargs.get("beta_a", 1e-3)
        self.alpha_y = kwargs.get("alpha_y", 1.)
        self.beta_y = kwargs.get("beta_y", 1e-3)

        self.gamma = kwargs.get("gamma", 1/32)
        self.theta = kwargs.get("theta", 1.)

        self.l_r = kwargs.get("l_r", 1e-3)

        self.mu = None
        self.phi = None

        self.b = None

        self.iterations = kwargs.get("iterations", None)

        self.latent_dim = kwargs.get("latent_dim", None)

        self.prng_handler = kwargs.get("prng_handler", None)

        self.optimizer = Adam(self.loss, **kwargs)

        self.kernel_fun = kwargs.get("kernel", None)

        self.ll_values = []

    def fit(self, y: jnp.ndarray, *args, **kwargs) -> None:
        """ Compute the Gaussian Process Dynamic Mode Decomposition given the data matrix
        :param y: Data matrix as jax numpy ndarray.
        """
        key = self.prng_handler.get_keys(1)
        variables = self._init_vars(key[0], y, self.latent_dim)

        y, self.lambda_y, x, lambda_x, lambda_0, a_tilde, lambda_a = self._train(variables)

        # Compute lower dimensional linear operator
        self.a_tilde = a_tilde.T

        # Compute DMD Values and DMD Modes
        self.mu, self.phi = jnp.linalg.eig(self.a_tilde)

        # Compute Amplitude of DMD Modes
        self.b = jnp.linalg.lstsq(self.phi, x.T[:, 0], rcond=None)[0]

    def predict(self, t_steps: jnp.ndarray, pow: float = 1) -> jnp.ndarray:
        """ Predict the data with the calculated DMD values
        :param t_steps: Time steps as jax.numpy.ndarray of size N
        :param pow: Exponential Power for vander matrix
        :return: Predicted Data as jax.numpy.ndarray
        """
        time_behaviour = self._vander(self.mu, len(t_steps), pow=pow)
        return (self.phi @ jnp.diag(self.b) @ time_behaviour).T

    def _train(self, variables):

        y, lambda_y, x, lambda_0, lambda_x, a_tilde, lambda_a = variables
        gamma, theta = self.gamma, self.theta

        opt_state = self.optimizer.opt_init([gamma, theta, x, lambda_y])

        y_outer = jnp.einsum('kd, hd -> kh', y, y)
        for _ in tqdm(range(self.iterations)):
            args = [y_outer, a_tilde, lambda_0, lambda_x, self.alpha_y, self.beta_y]
            params, opt_state, value = self.optimizer.step([gamma, theta, x, lambda_y],
                                                           opt_state, self.kernel_fun, *args)
            gamma, theta, x, lambda_y = params

            a_tilde = self._close_form_a(a_tilde, self.l_r, x, lambda_a, lambda_x)
            lambda_a = self._close_form_lambda_a(lambda_a, self.l_r, a_tilde, self.alpha_a, self.beta_a)
            lambda_0 = self._close_form_lambda_0(lambda_0, self.l_r, x, self.alpha_0, self.beta_0)
            lambda_x = self._close_form_lambda_x(lambda_x, self.l_r, x, a_tilde, self.alpha_x, self.beta_x)
            self.ll_values.append(value)

        self.gamma, self.theta = gamma, theta
        return [y, lambda_y, x, lambda_x, lambda_0, a_tilde, lambda_a]

    def get_mean_sigma(self, labels_x, labels_idx, t_steps):
        x = self.predict(t_steps)

        K11 = self.kernel_fun((self.gamma, self.theta), x[labels_idx, :].T, x[labels_idx, :].T)
        K12 = self.kernel_fun((self.gamma, self.theta), x[labels_idx, :].T, x.T)
        K21 = K12.T
        K22 = self.kernel_fun((self.gamma, self.theta), x.T, x.T)

        K11_inv = jnp.linalg.lstsq(K11 + jnp.eye(*K11.shape) / self.lambda_y, jnp.eye(K11.shape[0]))[0]

        return K21 @ K11_inv @ labels_x, K22 - K21 @ K11_inv @ K12

    def get_sample(self, labels_x, labels_idx, t_steps, number):

        mean, sigma = self.get_mean_sigma(labels_x, labels_idx, t_steps)

        mvnrml = MultivariateNormal(loc=mean.T, covariance_matrix=pos_def(sigma))
        return mvnrml.sample(self.prng_handler.get_keys(1)[0], sample_shape=(number,))

    def _init_vars(self, keys, y, m):
        """  This method initializes a sample given random keys, observations and the latent_dim

        :param keys (Jax device array): Pseudo random number generator (PRNG) Keys
        :param y (Jax devicearray): Contains a jax devicearray corresponding to the data
        :param m (int): Dimensionality of the latent space.
        :return:
        """
        t, _ = y.shape

        lambda_x = self.alpha_x / self.beta_x
        lambda_0 = self.alpha_0 / self.beta_0
        lambda_a = self.alpha_a / self.beta_a
        lambda_y = self.alpha_y / self.beta_y

        x = jnp.ones(shape=(t, m))
        a_tilde = random.normal(key=keys, shape=(m, m))
        return [y, lambda_y, x, lambda_0, lambda_x, a_tilde, lambda_a]

    @staticmethod
    @jit
    def _close_form_a(params, l_r, *args):
        x, lambda_a, lambda_x = args
        _, m = x.shape
        i = jnp.eye(m)
        a_tilde = jnp.linalg.lstsq(lambda_a / lambda_x * i + x[:-1, :].T @ x[:-1, :], i)[0] @ x[:-1, :].T @ x[1:, :]
        return params + l_r*(a_tilde - params)

    @staticmethod
    @jit
    def _close_form_lambda_a(params, l_r, *args):
        a_tilde, alpha_a, beta_a = args
        m = a_tilde.shape[0]

        lambda_a = (m**2 + 2*alpha_a - 2)/(jnp.trace(a_tilde @ a_tilde.T) + 2 * beta_a)
        return params + l_r*(lambda_a - params)

    @staticmethod
    @jit
    def _close_form_lambda_0(params, l_r, *args):
        x, alpha_0, beta_0 = args
        m = x.shape[1]

        lambda_0 = (m + 2 * alpha_0 - 2) / (jnp.trace(x[0:1, :].T@x[0:1, :]) + 2 * beta_0)
        return params + l_r*(lambda_0 - params)

    @staticmethod
    @jit
    def _close_form_lambda_x(params, l_r, *args):
        x, a_tilde, alpha_x, beta_x = args
        t = x.shape[0]
        m = a_tilde.shape[0]

        prod = jnp.trace(x[1:, :].T @ x[1:, :] - 2 * x[:-1, :].T @ x[1:, :] @ a_tilde.T
                         + x[:-1, :].T @ x[:-1, :] @ a_tilde @ a_tilde.T)
        lambda_x = ((t-1)*m + 2 * alpha_x - 2) / (prod + 2 * beta_x)
        return params + l_r*(lambda_x - params)

    @staticmethod
    @jax.partial(jit, static_argnums=(1,))
    def loss(params, kernel_fun, *args):
        gamma, theta, x, lambda_y = params
        y_outer, a_tilde, lambda_0, lambda_x, alpha_y, beta_y = args

        _, n = x.shape

        kernel = kernel_fun((gamma, theta), x.T, x.T)
        kernel += jnp.eye(*kernel.shape) / lambda_y
        chol = jnp.linalg.cholesky(kernel)
        cholinv = jnp.linalg.inv(chol)
        kernelinv = cholinv.T @ cholinv

        loss = 1 / 2 * (n * jnp.sum(jnp.log(jnp.linalg.det(kernel))) + jnp.einsum('nn -> ', kernelinv @ y_outer)
                        + lambda_x * jnp.trace(x[1:, :].T @ x[1:, :] - 2 * x[:-1, :].T @ x[1:, :] @ a_tilde.T
                                               + x[:-1, :].T @ x[:-1, :] @ a_tilde @ a_tilde.T)
                        + lambda_0*jnp.trace(x[0:1, :].T @x[0:1, :])
                        - (alpha_y - 1) * jnp.log(lambda_y) + beta_y*lambda_y)
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
