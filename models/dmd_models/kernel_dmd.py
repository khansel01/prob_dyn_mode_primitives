""" An implementation of kernel dynamic mode decomposition based.

References:
Kevrekidis, I. G., Rowley, C. W., & Williams, M. O. (2016). A kernel-based method for data-driven Koopman spectral
analysis. Journal of Computational Dynamics, 2(2), 247-265.

Kawahara, Y. (2016). Dynamic mode decomposition with reproducing kernels for Koopman spectral analysis. Advances in
neural information processing systems, 29, 911-919.

-----------------------------------------------------------------------------------------------------------------------
Several DMD methods in this project are inspired by the PyDMD package and have been transferred
to the jax library.

Reference:
Demo et al., (2018). PyDMD: Python Dynamic Mode Decomposition. Journal of Open Source Software, 3(22), 530
https://github.com/mathLab/PyDMD/tree/714ac2a9779890b16031d2c169e1eabdc46eeace

So if you're looking for a dynamic mode decomposition based on Python. Check out PyDMD!
-----------------------------------------------------------------------------------------------------------------------
"""

import jax.numpy as jnp
from jax.numpy import linalg
from models.dmd_models.base_dmd import BaseDMD


class KernelDMD(BaseDMD):
    def __init__(self, **kwargs):
        """ Kernelized Dynamic Mode Decomposition
        :param kernel: a certain kernel class from the kernel library
        """
        super().__init__()

        self.kernel = kwargs.get("kernel", None)
        if self.kernel is None:
            raise Warning(f"No Kernel selected. Please select a Kernel.")

        self.eig_fun = jnp.zeros(0)

    # override
    def fit(self, x0: jnp.ndarray, x1: jnp.ndarray, **kwargs) -> None:
        """ Compute the Total-Least_squares Dynamic Mode Decomposition given the two snapshot matrices x0 and x1
        :param x0: Snapshot matrix as jax numpy ndarray
        :param x1: Snapshot matrix as jax numpy ndarray
        """
        trunc_svd = kwargs.get("trunc_svd", 0.9999)

        # Compute the kernel matrices
        g = self.kernel.transform(x0, x0)
        a = self.kernel.transform(x1, x0)

        # SVD performs an eigh (sym. pos. def. matrix g) to get left singular vectors and singular values
        u, s = self._svd(g, trunc_svd)

        # Compute linear operator
        self.a_tilde = self._linear_op(a, u, s)

        # Compute DMD Values and DMD Modes and approximations of DMD eigenfunctions
        self.mu, self.phi, self.eig_fun = self._eig(self.a_tilde, x0, u, s)

        # Set Amplitudes of DMD Modes
        self.b = self.eig_fun[:, 0]

    # override
    @staticmethod
    def _linear_op(x: jnp.ndarray, *args) -> jnp.ndarray:
        """ Compute the kernel K_hat as linear operator.
        :param x: Kernel as jax.numpy.ndarray corresponding to kernel_func(x0, x1)
        :param args: Paramters could either be:
            args[0]: Left Singular Vectors U
            args[1]: Singular values S as jax.numpy.ndarray
        :return: Linear operator corresponds to Kernel k_hat as jax.numpy.ndarray
        """
        sigma_inv = jnp.diag(jnp.reciprocal(args[1]))
        return sigma_inv @ args[0].T @ x @ args[0] @ sigma_inv

    # override
    @staticmethod
    def _eig(lin_op: jnp.ndarray, *args) -> tuple:
        """ Compute the Kernel DMD Values and Kernel DMD Modes from the original system based on the lower dimensional
        linear operator.
        :param lin_op: lin_op as square matrix of type jax.numpy.ndarray
        :param args: Paramters could either be:
            args[0]: x snapshot matrix as jax.numpy.ndarray
            args[1]: Left Singular Vectors U as jax.numpy.ndarray
            args[2]: Singular values S as jax.numpy.ndarray
        :return: tuple containing the dmd values, the dmd modes and dmd eigenvalues each as jax.numpy.ndarray
        """
        mu, phi_lower = linalg.eig(lin_op)

        sigma_inv = jnp.diag(jnp.reciprocal(args[2]))

        phi = linalg.lstsq(phi_lower, sigma_inv @ args[1].T @ args[0].T, rcond=None)[0]

        eig_fun = args[1] @ jnp.diag(args[2]) @ phi_lower
        return mu, phi.T, eig_fun.T

    # override
    @staticmethod
    def _svd(x: jnp.ndarray, trunc_svd: float) -> tuple:
        """ Compute the left singular values and the singular values of the given kernel x. Instead of svd an eigen
        decomposition for hermitian matrices is used because x is a symmetric positive definite kernel.
        :param x: Kernel as jax.numpy.ndarray corresponding to kernel_func(x0, x0)
        :param trunc_svd: Describe different types of truncation.
            If ==0: Hard Threshold will be calculated
            elIf in [0, 1] of type float: Keep singular values representing Percentage of the data
            elIf >= 1 and of type int: Descirbes the truncation index
            else: no truncation is applied
        :return: Tuple containing left singular vector and singular values as jax.numpy.ndarray.
        """
        # get eigenvalues and eigenvectors
        s_2, u = linalg.eigh(x)

        # sort eigenvalues and eigenvectors
        idx = s_2.argsort()[::-1]
        u = u[:, idx]
        s_2 = s_2[idx]

        # Apply truncation
        if (trunc_svd > 0.) & (trunc_svd < 1.):
            trunc_idx = jnp.sum(jnp.cumsum(s_2) / jnp.sum(s_2) < trunc_svd)
        elif (trunc_svd >= 1) & isinstance(trunc_svd, int):
            trunc_idx = trunc_svd
        else:
            trunc_idx = s_2.shape[0]

        return u[:, :trunc_idx], jnp.sqrt(s_2[:trunc_idx])

