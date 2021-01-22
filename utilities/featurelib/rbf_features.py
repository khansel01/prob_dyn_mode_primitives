""" An implementation of RBF Features """

import jax
import jax.numpy as jnp
from jax import vmap, jit


class RBFFeatures(object):
    def __init__(self, center: jnp.array, gamma: float=1/32, theta: float=1.):
        """ Construct RBF features
        :param center: Set the centers of the rbf features as b x n x 1. B and n correspond to the number and the size
        of the centers, respectively.
        :param gamma: Describes the inverse width parameter of the feature
        """
        self.center = center
        self.gamma = gamma

    def transform(self, x: jnp.ndarray) -> jnp.ndarray:
        """ Transform given data into feature space based on radial basis function.
        :param x: Data matrix x as jax.numpy.ndarray
        :return: Normalized features as jax.numpy.ndarray
        """
        if x.ndim == 1:
            x = x[None, None, :]
        elif x.ndim == 2:
            x = x[None, :]
        elif x.ndim > 3:
            raise ValueError(f"The number of dimension of the input matrix has to be smaller or equal to three"
                             f". Dimension given x:{x.ndim}")
        inner = vmap(self._rbf_fun)(x - self.center)
        return inner / inner.sum(axis=0)

    @jax.partial(jit, static_argnums=(0,))
    def _rbf_fun(self, x: jnp.ndarray) -> jnp.ndarray:
        """ Calculate the RBF features
        :param x: Delta matrix x as jax.numpy.ndarray of size b x n x m.
        :return: Features as jax.numpy.ndarray
        """
        return jnp.exp(-self.gamma / 2 * jnp.diagonal(jnp.einsum('nj, ni-> ji', x, x))[None, :])
