""" An implementation of a probabilistic dynamic mode decomposition. Inspired bayesian dynamic mode decomposition based.
Instead of taking eigenvalues and eigenvectors from the posterior distributions, vectors of the linear operator
are sampled. Then, the exact DMD is used to obtain the DMD modes and DMD values.

References:
Takeishi, N., Kawahara, Y., Tabei, Y., & Yairi, T. (2017, August). Bayesian Dynamic Mode Decomposition.
In IJCAI (pp. 2814-2821).
https://github.com/thetak11/bayesiandmd

Tu, Jonathan H., et al. "On dynamic mode decomposition: Theory and applications." arXiv preprint arXiv:1312.0041 (2013).
https://arxiv.org/abs/1312.0041

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
from numpyro.distributions import Gamma, MultivariateNormal
from models.dmd_models.base_sampling_dmd import BaseSamplingDMD
from utils.utils_general import pos_def
from utils.gibbs_sampler import GibbsSampler


class ProbabilisticDMD(BaseSamplingDMD):
    def __init__(self, **kwargs):
        """ Probabilistic Dynamic Mode Decomposition

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
        conditionals = [self._conditional_a_tilde, self._conditional_fun_omega,
                        self._conditional_fun_z, self._conditional_fun_sigma2,
                        self._conditional_fun_nu2, self._conditional_fun_eta2]

        self.sampler = GibbsSampler(conditionals,
                                    iterations=kwargs.get("gibbs_iter", None),
                                    burn_in=kwargs.get("gibbs_burn_in", None),
                                    prng_handler=kwargs.get("prng_handler", None),
                                    ll_fun=self._log_likelihood)

    # override
    def fit(self, x0: jnp.ndarray, x1: jnp.ndarray, **kwargs) -> None:
        """ Compute the Probabilistic Dynamic Mode Decomposition given the two snapshot matrices x0 and x1

        :param x0: Snapshot matrix as jax numpy ndarray
        :param x1: Snapshot matrix as jax numpy ndarray
        """
        # Call sampling method
        self._sampling([x0, x1])

        # Get samples
        a_tilde_samples, omega_samples, z_samples, _, _, _ = self.samples

        # The linear operator is estimated by the mean value of a_tilde_samples
        self.a_tilde = jnp.sum(jnp.array(a_tilde_samples), axis=0) / len(a_tilde_samples)

        # Compute DMD Values and DMD Modes
        omega = jnp.sum(jnp.array(omega_samples), axis=0) / len(omega_samples)
        self.mu, self.phi = self._eig(self.a_tilde, omega)

        # Compute Amplitude of DMD Modes
        z = jnp.sum(jnp.array(z_samples), axis=0)[:, 0] / len(z_samples)
        self.b = self._amplitude(omega @ z, self.phi)

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
        a_tilde = 2 * random.uniform(keys[0], (latent_dim, latent_dim)) - 1
        omega = 2 * random.uniform(keys[1], (n, latent_dim)) - 1
        z = 2 * random.uniform(keys[2], (latent_dim, m)) - 1

        # initialize hyper parameters
        nu2 = jnp.ones((n, latent_dim))
        eta2 = jnp.ones((latent_dim, latent_dim))

        # MLE for initial sigma2
        f1 = x0 - omega @ z
        f2 = x1 - omega @ a_tilde @ z
        sigma2 = jnp.sum(jnp.conj(f1) * f1 + jnp.conj(f2) * f2) / (2 * n * m - 1)
        return [a_tilde, omega, z, sigma2, nu2, eta2]

    # -----------------------------------------------------------------------------------------------------------------
    # Next are the conditionals and the log-likelihood function for the Gibbs sampler
    # -----------------------------------------------------------------------------------------------------------------
    @staticmethod
    @jax.partial(jit, static_argnums=(1,))
    def _conditional_a_tilde(keys, observations, sample, *args) -> tuple:
        """ A_tilde's Conditional function based on a Multivariate Gaussian distribution.

        :param keys: Pseudo random number generator (PRNG) Keys
        :param observations: Contains two jax devicearray corresponding to the two snapshot matrices
        :param sample: Contains the following parameters a_tilde, omega, z, sigma2, nu2 and eta2.
        :param args: None
        :return: tuple contains:
            Pseudo random number generator (PRNG) Keys
            A new sample a_tilde, omega, z, sigma2, nu2 and eta2.
        """
        _, x1 = observations
        a_tilde, omega, z, sigma2, nu2, eta2 = sample

        latent_dim = a_tilde.shape[0]

        for l_iter in range(latent_dim):
            _idx = list(range(l_iter)) + list(range(l_iter + 1, latent_dim))
            _sigma_inv = jnp.diag(1 / eta2[:, l_iter]) + \
                ((omega.conj().T @ omega) * (z[l_iter].conj().T @ z[l_iter])) / sigma2
            _sigma = jnp.linalg.inv(_sigma_inv)

            _mu = _sigma @ ((omega.conj().T @ (
                    x1 - omega @ (a_tilde[:, _idx] @ z[_idx]))) @ z[l_iter].conj().T) / sigma2

            _normal = MultivariateNormal(loc=_mu, covariance_matrix=pos_def(_sigma))
            a_tilde = index_update(a_tilde, index[:, l_iter], _normal.sample(keys[l_iter]))

        return keys[latent_dim:], [a_tilde, omega, z, sigma2, nu2, eta2]

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
            A new sample a_tilde, omega, z, sigma2, nu2 and eta2.
        """
        x0, x1 = observations
        a_tilde, omega, z, sigma2, nu2, eta2 = sample

        latent_dim = a_tilde.shape[0]

        for l_iter in range(latent_dim):
            _idx = list(range(l_iter)) + list(range(l_iter + 1, latent_dim))
            _sigma_inv = jnp.diag(1 / nu2[:, l_iter]) + (
                    z[l_iter].conj().T @ z[l_iter] + (a_tilde[l_iter, :] @ z) @ (
                    z.conj().T @ a_tilde.conj().T[:, l_iter])) * jnp.eye(x0.shape[0]) / sigma2
            _sigma = jnp.linalg.inv(_sigma_inv)

            _mu = _sigma @ ((x0 - omega[:, _idx] @ z[_idx]) @ z[l_iter].conj().T +
                            ((x1 - omega[:, _idx] @ (a_tilde @ z)[_idx]) @
                             (z.conj().T @ a_tilde[l_iter].conj().T))) / sigma2

            _normal = MultivariateNormal(loc=_mu, covariance_matrix=pos_def(_sigma))
            omega = index_update(omega, index[:, l_iter], _normal.sample(keys[l_iter]))

        return keys[latent_dim:], [a_tilde, omega, z, sigma2, nu2, eta2]

    @staticmethod
    def _conditional_fun_z(keys, observations, sample, *args) -> tuple:
        """ Z's Conditional function based on a Multivariate Gaussian distribution.

        :param keys: Pseudo random number generator (PRNG) Keys
        :param observations: Contains two jax devicearray corresponding to the two snapshot matrices
        :param sample: Contains the following parameters a_tilde, omega, z, sigma2, nu2 and eta2.
        :param args: None
        :return: tuple contains:
            Pseudo random number generator (PRNG) Keys
            A new sample a_tilde, omega, z, sigma2, nu2 and eta2.
        """
        x0, x1 = observations
        a_tilde, omega, z, sigma2, nu2, eta2 = sample

        latent_dim = a_tilde.shape[0]

        sigma_inv = jnp.eye(latent_dim) + (omega.conj().T @ omega +
                                           (a_tilde.conj().T @ omega.conj().T) @ (omega @ a_tilde)) / sigma2
        _lambda, _u = jnp.linalg.eig(0.5 * (sigma_inv + sigma_inv.conj().T))  # ensure Hermitian
        _sigma = jnp.real(_u @ jnp.diag(jnp.reciprocal(_lambda)) @ _u.conj().T)

        _mu = _sigma @ (omega.conj().T @ x0 + a_tilde.conj().T @ omega.conj().T @ x1) / sigma2

        _normal = MultivariateNormal(loc=_mu.T, covariance_matrix=pos_def(_sigma))
        return keys[1:], [a_tilde, omega, _normal.sample(keys[0]).T, sigma2, nu2, eta2]

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
            A new sample a_tilde, omega, z, sigma2, nu2 and eta2.
        """
        x0, x1 = observations
        a_tilde, omega, z, sigma2, nu2, eta2 = sample
        alpha, beta = args

        n, m = x0.shape
        f1 = x0 - omega @ z
        f2 = x1 - omega @ a_tilde @ z

        gamma = Gamma(concentration=alpha + 2 * n * m,
                      rate=beta + jnp.sum(jnp.conj(f1) * f1) + jnp.sum(jnp.conj(f2) * f2))
        return keys[1:], [a_tilde, omega, z, 1/gamma.sample(keys[0]), nu2, eta2]

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
            A new sample a_tilde, omega, z, sigma2, nu2 and eta2.
        """
        a_tilde, omega, z, sigma2, nu2, eta2 = sample
        alpha, beta = args

        gamma = Gamma(concentration=alpha + 1, rate=beta + omega.conj() * omega)
        return keys[1:], [a_tilde, omega, z, sigma2, 1/gamma.sample(keys[0]), eta2]

    @staticmethod
    def _conditional_fun_eta2(keys, observations, sample, *args) -> tuple:
        """ Eta2's Conditional function based on a Inverse Gamma distribution.

        :param keys: Pseudo random number generator (PRNG) Keys
        :param observations: Contains two jax devicearray corresponding to the two snapshot matrices
        :param sample: Contains the following parameters a_tilde, omega, z, sigma2, nu2 and eta2.
        :param args:
            alpha (float>0) : Initial shape of the gamma distribution.
            beta (float>0) : Initial rate of the gamma distribution.
        :return: tuple contains:
            Pseudo random number generator (PRNG) Keys
            A new sample a_tilde, omega, z, sigma2, nu2 and eta2.
        """
        a_tilde, omega, z, sigma2, nu2, eta2 = sample
        alpha, beta = args

        gamma = Gamma(concentration=alpha + 1, rate=beta + a_tilde.conj() * a_tilde)
        return keys[1:], [a_tilde, omega, z, sigma2, nu2, 1/gamma.sample(keys[0])]

    @staticmethod
    def _log_likelihood(observations, sample):
        """ Calculate the log likelihood value based on the observations and the given sample.

        :param observations (list): Contains two jax device array corresponding to the two snapshot matrices
        :param sample (list): Contains the following parameters a_tilde, omega, z, sigma2, nu2 and eta2.
        :return: Log-likelihood value as jax device array.
        """
        x0, x1 = observations
        a_tilde, omega, z, sigma2, _, _ = sample

        n, m = x0.shape

        f1 = x0 - omega @ z  # N x M
        f2 = x1 - omega @ a_tilde @ z  # N x M
        return - 2 * n * m * jnp.log(sigma2 * jnp.pi) - \
            (jnp.sum(jnp.conj(f1) * f1) + jnp.sum(jnp.conj(f2) * f2)) / sigma2



