""" An implementation of a polynomial Kernel (gamma * x.T @ y + theta) ** pow
"""

import jax.numpy as jnp


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
            inner = jnp.einsum('n, n -> ', x, y)
        elif x.ndim == 2 & y.ndim == 2:
            inner = jnp.einsum('ni, nj -> ij', x, y)
        elif x.ndim == 3 & y.ndim == 3:
            inner = jnp.einsum('bni, bnj -> bij', x, y)
        else:
            raise ValueError(f"The inputs have to be the same dimension. Dimension given x:{x.ndim}, y:{y.ndim}")

        return (self.gamma * inner + self.theta) ** self.deg
