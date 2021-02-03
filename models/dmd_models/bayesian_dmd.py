""" An implementation of bayesian dynamic mode decomposition based.

Reference:
Takeishi, N., Kawahara, Y., Tabei, Y., & Yairi, T. (2017, August). Bayesian Dynamic Mode Decomposition.
In IJCAI (pp. 2814-2821).
https://github.com/thetak11/bayesiandmd
-----------------------------------------------------------------------------------------------------------------------
Several DMD methods in this project are inspired by the PyDMD package and have been transferred
to the jax library.

Reference:
Demo et al., (2018). PyDMD: Python Dynamic Mode Decomposition. Journal of Open Source Software, 3(22), 530
https://github.com/mathLab/PyDMD/tree/714ac2a9779890b16031d2c169e1eabdc46eeace

So if you're looking for a dynamic mode decomposition based on Python. Check out PyDMD!
-----------------------------------------------------------------------------------------------------------------------
"""

import jax
import jax.numpy as jnp
import jax.random as random

from jax import jit
from jax.ops import index, index_update
from numpyro.distributions import Gamma
from models.dmd_models.base_sampling_dmd import BaseSamplingDMD
from utilities.gibbs_sampler import GibbsSampler
from utilities.utils import sample_complex_normal


class BayesianDMD(BaseSamplingDMD):
    def __init__(self, **kwargs):
        """ Bayesian Dynamic Mode Decomposition

        :param kwargs:
            alpha (float>0) : Initial shape of the gamma distribution.
            beta (float>0) : Initial rate of the gamma distribution.
            latent_dim (int>0) : Dimensionality of the latent space.
            prng_handler(class) : Class that handles the pseudo random number generation
            gibbs_iter(int) : Specifies the iterations of the Gibbs sampler
            gibbs_burn_in (int) : Specifies the Burn-in phase of the Gibbs sampler
        """
        super().__init__(**kwargs)

        # Initialize Gibbs_sampler
        conditionals = [self._conditional_fun_lambda, self._conditional_fun_omega,
                        self._conditional_fun_z, self._conditional_fun_sigma2,
                        self._conditional_fun_nu2]

        self.sampler = GibbsSampler(conditionals,
                                    iterations=kwargs.get("gibbs_iter", None),
                                    burn_in=kwargs.get("gibbs_burn_in", None),
                                    prng_handler=kwargs.get("prng_handler", None),
                                    ll_fun=self._log_likelihood)

    # override
    def fit(self, x0: jnp.ndarray, x1: jnp.ndarray, **kwargs) -> None:
        """ Compute the Bayesian Dynamic Mode Decomposition given the two snapshot matrices x0 and x1

        :param x0: Snapshot matrix as jax numpy ndarray
        :param x1: Snapshot matrix as jax numpy ndarray
        """
        # Call sampling method
        self._sampling([x0, x1])

        # Get samples
        _lambda_samples, omega_samples, z_samples, _, _ = self.samples

        self.mu = jnp.sum(jnp.array(_lambda_samples), axis=0).T / len(_lambda_samples)

        # Compute DMD Values and DMD Modes
        omega = jnp.sum(jnp.array(omega_samples), axis=0) / len(omega_samples)
        self.phi = omega

        # Compute Amplitude of DMD Modes
        z = jnp.sum(jnp.array(z_samples), axis=0)[:, 0] / len(z_samples)
        self.b = z

    # -----------------------------------------------------------------------------------------------------------------
    # Next are methods associated with the initialization of the gibbs sampler
    # -----------------------------------------------------------------------------------------------------------------
    @staticmethod
    def _sample_init(keys, observations, latent_dim):
        """  This method initializes a sample given random keys, observations and the latent_dim

        :param keys (Jax device array): Pseudo random number generator (PRNG) Keys
        :param observations (list): Contains two jax devicearray corresponding to the two snapshot matrices
        :param latent_dim (int): Dimensionality of the latent space.
        :return:
        """
        x0, x1 = observations

        # Get shapes
        n, m = x0.shape

        # initialize random variables and hyper parameters
        _lambda = 2 * random.uniform(keys[0], (1, latent_dim)) - 1 +\
            1j * (2 * random.uniform(keys[1], (1, latent_dim)) - 1)

        omega = 2 * random.uniform(keys[2], (n, latent_dim)) - 1 +\
            1j * (2 * random.uniform(keys[3], (n, latent_dim)) - 1)

        z = 2 * random.uniform(keys[4], (latent_dim, m)) - 1 +\
            1j * (2 * random.uniform(keys[5], (latent_dim, m)) - 1)  # N x K

        # initialize hyper parameters
        nu2 = jnp.ones((n, latent_dim))

        # MLE for initial sigma2
        f1 = x0 - omega @ z
        f2 = x1 - omega @ jnp.diag(*_lambda) @ z
        sigma2 = jnp.sum(jnp.conj(f1) * f1 + jnp.conj(f2) * f2) / (2 * n * m - 1)
        return [_lambda, omega, z, sigma2, nu2]

    # -----------------------------------------------------------------------------------------------------------------
    # Next are the conditionals and the log-likelihood function for the Gibbs sampler
    # -----------------------------------------------------------------------------------------------------------------
    @staticmethod
    @jax.partial(jit, static_argnums=(1,))
    def _conditional_fun_lambda(keys, observations, sample, *args) -> tuple:
        """ Lambda's Conditional function based on a Multivariate Gaussian distribution.

        :param keys: Pseudo random number generator (PRNG) Keys
        :param observations: Contains two jax devicearray corresponding to the two snapshot matrices
        :param sample: Contains the following parameters a_tilde, omega, z, sigma2, nu2 and eta2.
        :param args: None
        :return: tuple contains:
            Pseudo random number generator (PRNG) Keys
            A new sample _lambda, omega, z, sigma2, nu2.
        """
        _, x1 = observations
        _lambda, omega, z, sigma2, nu2 = sample

        latent_dim = _lambda.shape[1]
        for l_iter in range(latent_dim):
            _idx = list(range(l_iter)) + list(range(l_iter + 1, latent_dim))

            eta = omega[:, _idx] @ jnp.diag(*_lambda[:, _idx]) @ z[_idx, :]

            _sigma_inv = jnp.ones(1) + \
                (jnp.conj(omega[:, l_iter]) @ omega[:, l_iter].T) * (z[l_iter, :].conj().T @ z[l_iter, :]) / sigma2

            _mu = jnp.conj(omega[:, l_iter]) @ jnp.sum((x1 - eta) * jnp.conj(z[l_iter, :]), axis=1) / \
                (sigma2 * _sigma_inv)

            _lambda = index_update(_lambda, index[:, l_iter],
                                   sample_complex_normal(keys[l_iter], _mu, jnp.reciprocal(_sigma_inv))[0])

        return keys[latent_dim:], [_lambda, omega, z, sigma2, nu2]

    @staticmethod
    @jax.partial(jit, static_argnums=(1,))
    def _conditional_fun_omega(keys, observations, sample, *args) -> tuple:
        """ Omega's Conditional function based on a Multivariate Gaussian distribution.

        :param keys: Pseudo random number generator (PRNG) Keys
        :param observations: Contains two jax devicearray corresponding to the two snapshot matrices
        :param sample: Contains the following parameters a_tilde, omega, z, sigma2, nu2 and eta2.
        :param args: None
        :return: tuple contains:
            Pseudo random number generator (PRNG) Keys
            A new sample _lambda, omega, z, sigma2, nu2.
        """
        x0, x1 = observations
        _lambda, omega, z, sigma2, nu2 = sample

        n, m = x0.shape
        latent_dim = _lambda.shape[1]

        for l_iter in range(latent_dim):
            _idx = list(range(l_iter)) + list(range(l_iter + 1, latent_dim))

            xi = omega[:, _idx] @ z[_idx, :]
            eta = omega[:, _idx] @ jnp.diag(*_lambda[:, _idx]) @ z[_idx, :]

            _sigma_inv = (1 + jnp.conj(_lambda[:, l_iter]) @ _lambda[:, l_iter]) / sigma2 * \
                z[l_iter, :].conj().T @ z[l_iter, :] * jnp.eye(n) + jnp.diag(jnp.reciprocal(1 / nu2.T[l_iter]))
            _sigma = jnp.diag(jnp.reciprocal(jnp.diag(_sigma_inv)))

            _mu = (jnp.sum((x0 - xi) * jnp.conj(z[l_iter, :]), axis=1) + jnp.conj(
                _lambda[:, l_iter]) * jnp.sum((x1 - eta) * jnp.conj(z[l_iter, :]), axis=1)) / sigma2 @ _sigma

            omega = index_update(omega, index[:, l_iter], sample_complex_normal(keys[l_iter], _mu, _sigma)[0])

        return keys[latent_dim:], [_lambda, omega, z, sigma2, nu2]

    @staticmethod
    def _conditional_fun_z(keys, observations, sample, *args) -> tuple:
        """ Z's Conditional function based on a Multivariate Gaussian distribution.

        :param keys: Pseudo random number generator (PRNG) Keys
        :param observations: Contains two jax devicearray corresponding to the two snapshot matrices
        :param sample: Contains the following parameters a_tilde, omega, z, sigma2, nu2 and eta2.
        :param args: None
        :return: tuple contains:
            Pseudo random number generator (PRNG) Keys
            A new sample _lambda, omega, z, sigma2, nu2.
        """
        x0, x1 = observations
        _lambda, omega, z, sigma2, nu2 = sample

        latent_dim = _lambda.shape[1]

        _sigma_inv = jnp.conj(omega.T) @ omega
        _sigma_inv += jnp.diag(jnp.conj(*_lambda)) @ jnp.conj(omega.T) @ omega @ jnp.diag(*_lambda)
        _sigma_inv /= sigma2
        _sigma_inv += jnp.eye(latent_dim)
        _sigma_inv = 0.5 * (_sigma_inv + _sigma_inv.conj().T)  # ensure Hermitian
        lambda_phi, u_phi = jnp.linalg.eig(_sigma_inv)
        _sigma = u_phi @ jnp.diag(jnp.reciprocal(lambda_phi)) @ u_phi.conj().T

        _mu = _sigma @ (jnp.conj(omega).T @ x0 + jnp.diag(jnp.conj(*_lambda)) @ jnp.conj(omega).T @ x1) / sigma2

        sample = sample_complex_normal(keys[0], _mu, _sigma)
        return keys[1:], [_lambda, omega, sample.T, sigma2, nu2]

    @staticmethod
    def _conditional_fun_sigma2(keys, observations, sample, *args) -> tuple:
        """ Sigma2's Conditional function based on a Inverse Gamma distribution.

        :param keys: Pseudo random number generator (PRNG) Keys
        :param observations: Contains two jax devicearray corresponding to the two snapshot matrices
        :param sample: Contains the following parameters a_tilde, omega, z, sigma2, nu2 and eta2.
        :param args:
            alpha (float>0) : Initial shape of the gamma distribution.
            beta (float>0) : Initial rate of the gamma distribution.
        :return: tuple contains:
            Pseudo random number generator (PRNG) Keys
            A new sample _lambda, omega, z, sigma2, nu2.
        """
        x0, x1 = observations
        _lambda, omega, z, sigma2, nu2 = sample
        alpha, beta = args

        n, m = x0.shape

        f1 = x0 - omega @ z
        f2 = x1 - omega @ jnp.diag(*_lambda) @ z
        gamma = Gamma(concentration=alpha + 2 * n * m,
                      rate=beta + jnp.sum(jnp.conj(f1) * f1) + jnp.sum(jnp.conj(f2) * f2))
        return keys[1:], [_lambda, omega, z, 1/gamma.sample(keys[0]), nu2]

    @staticmethod
    def _conditional_fun_nu2(keys, observations, sample, *args) -> tuple:
        """ Nu2's Conditional function based on a Inverse Gamma distribution.

        :param keys: Pseudo random number generator (PRNG) Keys
        :param observations: Contains two jax devicearray corresponding to the two snapshot matrices
        :param sample: Contains the following parameters a_tilde, omega, z, sigma2, nu2 and eta2.
        :param args:
            alpha (float>0) : Initial shape of the gamma distribution.
            beta (float>0) : Initial rate of the gamma distribution.
        :return: tuple contains:
            Pseudo random number generator (PRNG) Keys
            A new sample _lambda, omega, z, sigma2, nu2.
        """
        _lambda, omega, z, sigma2, nu2 = sample
        alpha, beta = args

        gamma = Gamma(concentration=alpha + 1, rate=beta + jnp.conj(omega) * omega)
        return keys[1:], [_lambda, omega, z, sigma2, 1/gamma.sample(keys[0])]

    @staticmethod
    def _log_likelihood(observations, sample):
        """ Calculate the log likelihood value based on the observations and the given sample.

        :param observations (list): Contains two jax device array corresponding to the two snapshot matrices
        :param sample (list): Contains the following parameters a_tilde, omega, z, sigma2, nu2 and eta2.
        :return: Log-likelihood value as jax device array.
        """
        x0, x1 = observations
        _lambda, omega, z, sigma2, nu2 = sample

        n, m = x0.shape

        f1 = x0 - omega @ z  # N x M
        f2 = x1 - omega @ jnp.diag(*_lambda) @ z  # N x M
        return - 2 * n * m * jnp.log(sigma2 * jnp.pi) - \
            (jnp.sum(jnp.conj(f1) * f1) + jnp.sum(jnp.conj(f2) * f2)) / sigma2
