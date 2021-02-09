""" Python script containing all kernel functions used in this project

------------------------------------------------------------------------------------------------------------------------
Already implemented kernels:

1. Polynomial Kernel.
2. RBF Kernel.

------------------------------------------------------------------------------------------------------------------------
"""

import jax
import jax.numpy as jnp
from jax import vmap, jit


class BaseKernel(object):
    def __init__(self):
        """ Construct a the Basis Kernel class """

    def transform(self, x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
        """ Transform given data into a symmetric positive definite Kernel
        :param x: First matrix x as jax.numpy.ndarray
        :param y: Second matrix x as jax.numpy.ndarray
        :return: A positive definite Kernel as jax.numpy.ndarray
        """
        if x.ndim == 1 & y.ndim == 1:
            return self._kernel_fun(x[:, None], y[:, None])
        elif x.ndim == 2 & y.ndim == 2:
            return self._kernel_fun(x, y)
        elif x.ndim == 3 & y.ndim == 3:
            return vmap(self._kernel_fun)(x, y)
        else:
            raise ValueError(f"The inputs have to be the same dimension. Dimension given x:{x.ndim}, y:{y.ndim}")

    @jax.partial(jit, static_argnums=(0,))
    def _kernel_fun(self, *args):
        """ Calculate a predefined kernel density function.
        """
        raise NotImplementedError(f"No Kernel function implemented in Base class.")


class PolynomialKernel(BaseKernel):
    def __init__(self, gamma: float = 1., theta: float = 1 / 16., deg: int = 1):
        """ Construct a PolynomialKernel of the form (gamma * x.T @ y + th   eta) ** pow
        :param gamma: The process variance controls the weight of the outer product as float
        :param theta: bias of the kernel as float. If zero => homogeneous kernel
        :param pow: describes the degree of the polynomial kernel.
        """
        super().__init__()
        self.gamma = gamma
        self.theta = theta
        self.deg = deg

    # override
    @jax.partial(jit, static_argnums=(0,))
    def _kernel_fun(self, x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
        """ Calculate a Polynomial Kernel
        :param x: First matrix x as jax.numpy.ndarray
        :param y: Second matrix x as jax.numpy.ndarray
        :return:A positive definite RBF Kernel
        """
        return (self.gamma * jnp.einsum('ni, nj -> ij', x, y) + self.theta) ** self.deg


class RBFKernel(BaseKernel):
    def __init__(self, gamma: float=1/32, theta: float=1.):
        """ Construct a Radial Basis function Kernel
        :param gamma: Describes the inverse width parameter of the kernel
        :param theta: bias of the kernel as float. If zero => homogeneous kernel
        """
        super().__init__()
        self.gamma = gamma
        self.theta = theta

    # override
    @jax.partial(jit, static_argnums=(0,))
    def _kernel_fun(self, x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
        """ Calculate a RBF Kernel
        :param x: First matrix x as jax.numpy.ndarray
        :param y: Second matrix x as jax.numpy.ndarray
        :return:A positive definite RBF Kernel
        """
        i = x.shape[1]
        j = y.shape[1]
        inner = - 2 * jnp.einsum('ni, nj -> ij', x, y) + \
            jnp.ones((i, j)) * jnp.diag(jnp.einsum('ni, nj -> ij', y, y)) + \
            (jnp.ones((j, i)) * jnp.diag(jnp.einsum('ni, nj -> ij', x, x))).T
        return self.theta * jnp.exp(-self.gamma/2 * inner)
