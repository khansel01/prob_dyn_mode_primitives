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
from jax.ops import index, index_update
from numpyro.distributions import Gamma
from models.dmd_models.base_sampling_dmd import BaseSamplingDMD
from utilities.gibbs_sampler import GibbsSampler
from utilities.utils import sample_complex_normal


class GPDMD(object):
    def __init__(self):
        """ Dynamic Mode Decomposition """
        self.mu = jnp.zeros(0)
        self.phi = jnp.zeros(0)
        self.a_tilde = jnp.eye(2)
        self.b = jnp.zeros(0)

    def fit(self, x: jnp.ndarray, *args, **kwargs) -> None:
        """ Compute the Total-Least_squares Dynamic Mode Decomposition given the two snapshot matrices x0 and x1
        :param x: Snapshot matrix as jax numpy ndarray
        """
        x0, x1 = x0.T, x1.T

        raise NotImplementedError(f"No fit() method available." 
                                  "BaseDMD only specifies the default construct of the corresponding DMD methods.")

    def predict(self, t_steps: jnp.ndarray, pow: float = 1) -> jnp.ndarray:
        """ Predict the data with the calculated DMD values
        :param t_steps: Time steps as jax.numpy.ndarray of size N
        :param pow: Exponential Power for vander matrix
        :return: Predicted Data as jax.numpy.ndarray
        """
        time_behaviour = self._vander(self.mu, len(t_steps), pow=pow)
        return self.phi @ jnp.diag(self.b) @ time_behaviour

    @staticmethod
    def _linear_op(x: jnp.ndarray, *args) -> jnp.ndarray:
        """ Compute the lower dimensional linear operator of the underlying dynamics in the system.
        :param x: Snapshot matrix as jax.numpy.ndarray
        :param args: Paramters could either be:
            args[0]: x1 snapshot matrix as jax.numpy.ndarray
            or:
            args[0]: Left singular vectors U as jax.numpy.ndarray
            args[1]: Singular values S as jax.numpy.ndarray
            args[2]: Right singular Vectors V as jax.numpy.ndarray
        :return: linear operator as jax.numpy.ndarray
        """
        if len(args) == 1:
            _u, _, _ = jnp.linalg.svd(jnp.append(x, args[0], axis=0), full_matrices=False)
            trunc_svd = len(_u)//2
            return _u[trunc_svd:, :trunc_svd] @ jnp.linalg.inv(_u[:trunc_svd, :trunc_svd])
        elif len(args) == 3:
            return args[0].conj().T @ x @ args[2] @ jnp.diag(jnp.reciprocal(args[1]))
        else:
            raise ValueError