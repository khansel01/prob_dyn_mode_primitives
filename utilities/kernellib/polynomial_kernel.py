""" An implementation of a polynomial Kernel (gamma * x.T @ y + theta) ** pow """

import jax
import jax.numpy as jnp
from jax import vmap, jit


class PolynomialKernel(object):
    def __init__(self, gamma: float=1., theta: float=1/16., deg: int=1):
        """ Construct a PolynomialKernel of the form (gamma * x.T @ y + theta) ** pow
        :param gamma: The process variance controls the weight of the outer product as float
        :param theta: bias of the kernel as float. If zero => homogeneous kernel
        :param pow: describes the degree of the polynomial kernel.
        """
        self.gamma = gamma
        self.theta = theta
        self.deg = deg

    def transform(self, x, y) -> jnp.ndarray:
        """ Transform given data into a symmetric positive definite polynomial Kernel
        :param x: First matrix x as jax.numpy.ndarray
        :param y: Second matrix x as jax.numpy.ndarray
        :return: A positive definite polynomial Kernel
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
    def _kernel_fun(self, x, y):
        """ Calculate a Polynomial Kernel
        :param x: First matrix x as jax.numpy.ndarray
        :param y: Second matrix x as jax.numpy.ndarray
        :return:A positive definite RBF Kernel
        """
        inner = jnp.einsum('ni, nj -> ij', x, y)
        return (self.gamma * inner + self.theta) ** self.deg
