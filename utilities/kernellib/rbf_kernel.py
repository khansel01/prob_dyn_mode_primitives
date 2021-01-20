""" An implementation of a Radial Basis function Kernel """

import jax
import jax.numpy as jnp
from jax import vmap, jit


class RBFKernel(object):
    def __init__(self, gamma: float=1/32, theta: float=1.):
        """ Construct a Radial Basis function Kernel
        :param gamma: Describes the inverse width paramter of the kernel
        :param theta: bias of the kernel as float. If zero => homogeneous kernel
        """
        self.gamma = gamma
        self.theta = theta

    def transform(self, x, y) -> jnp.ndarray:
        """ Transform given data into a symmetric positive definite RBF Kernel
        :param x: First matrix x as jax.numpy.ndarray
        :param y: Second matrix x as jax.numpy.ndarray
        :return:A positive definite RBF Kernel
        """
        if x.ndim == 1 & y.ndim == 1:
            return self._kernel_fun(x.reshape(-1, 1), y.reshape(-1, 1))
        elif x.ndim == 2 & y.ndim == 2:
            return self._kernel_fun(x, y)
        elif x.ndim == 3 & y.ndim == 3:
            return vmap(self._kernel_fun)(x, y)
        else:
            raise ValueError(f"The inputs have to be the same dimension. Dimension given x:{x.ndim}, y:{y.ndim}")

    @jax.partial(jit, static_argnums=(0,))
    def _kernel_fun(self, x, y):
        """ Calculate a RBF Kernel
        :param x: First matrix x as jax.numpy.ndarray
        :param y: Second matrix x as jax.numpy.ndarray
        :return:A positive definite RBF Kernel
        """
        inner = - 2 * jnp.einsum('ni, nj -> ij', x, y)
        inner += jnp.ones_like(inner) * jnp.diag(jnp.einsum('ni, nj -> ij', y, y))
        inner += (jnp.ones_like(inner) * jnp.diag(jnp.einsum('ni, nj -> ij', x, x))).T
        return self.theta * jnp.exp(-self.gamma/2 * inner)
