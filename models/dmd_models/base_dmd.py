""" The Python script Base DMD provides the default construct for all further DMD scripts.

-----------------------------------------------------------------------------------------------------------------------
Most DMD methods in this project including BaseDMD are inspired by the PyDMD package and have been transferred
to the jax library.

Reference:
Demo et al., (2018). PyDMD: Python Dynamic Mode Decomposition. Journal of Open Source Software, 3(22), 530
https://github.com/mathLab/PyDMD/tree/714ac2a9779890b16031d2c169e1eabdc46eeace

So if you're looking for a dynamic mode decomposition based on Python. Check out PyDMD!
-----------------------------------------------------------------------------------------------------------------------
"""

import jax.numpy as jnp
from jax.numpy import linalg


class BaseDMD(object):
    def __init__(self):
        """ Dynamic Mode Decomposition """
        self.mu = jnp.zeros(0)
        self.phi = jnp.zeros(0)
        self.a_tilde = jnp.eye(2)
        self.b = jnp.zeros(0)

    def fit(self, x0: jnp.ndarray, x1: jnp.ndarray, *args, **kwargs) -> None:
        """ Compute the Total-Least_squares Dynamic Mode Decomposition given the two snapshot matrices x0 and x1
        :param x0: Snapshot matrix as jax numpy ndarray
        :param x1: Snapshot matrix as jax numpy ndarray
        """
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
        :return:
        """
        if len(args) == 1:
            _u, _, _ = linalg.svd(jnp.append(x, args[0], axis=0), full_matrices=False)
            trunc_svd = len(_u)//2
            return _u[trunc_svd:, :trunc_svd] @ linalg.inv(_u[:trunc_svd, :trunc_svd])
        elif len(args) == 3:
            return args[0].conj().T @ x @ args[2] @ jnp.diag(jnp.reciprocal(args[1]))
        else:
            raise ValueError

    @staticmethod
    def _eig(lin_op: jnp.ndarray, *args) -> tuple:
        """ Compute the DMD Values and DMD Modes from the original system based on the lower dimensional linear
            operator.
        :param lin_op: lin_op as square matrix of type jax.numpy.ndarray
        :param args: Parameters could either be:
            args[0]: Left singular vectors U as jax.numpy.ndarray
            or:
            args[0]: x1 snapshot matrix as jax.numpy.ndarray
            args[1]: Singular values S as jax.numpy.ndarray
            args[2]: Right singular Vectors V as jax.numpy.ndarray
        :return: tuple containing the dmd values and the dmd modes each as jax.numpy.ndarray
        """
        mu, phi_lower = linalg.eig(lin_op)

        if len(args) == 1:
            return mu, args[0] @ phi_lower
        elif len(args) == 3:
            return mu, args[0] @ args[2] @ jnp.diag(jnp.reciprocal(args[1])) @ phi_lower @ jnp.diag(jnp.reciprocal(mu))
        else:
            raise ValueError

    @staticmethod
    def _svd(x: jnp.ndarray, trunc_svd: float) -> tuple:
        """ Compute the lower dimensional Singual Value Decomposition
        :param x: snapshot matrix as jax.numpy.ndarray
        :param trunc_svd: Describe different types of truncation.
            If ==0: Hard Threshold will be calculated
            elIf in [0, 1] of type float: Keep singular values representing Percentage of the data
            elIf >= 1 and of type int: Descirbes the truncation index
            else: no truncation is applied
        :return: Tuple containing singular values, right and the singular vectors each as jax.numpy.ndarray.

        References:
        Gavish, M., & Donoho, D. L. (2014). The optimal hard threshold for singular values is 4/sqrt(3).
        IEEE Transactions on Information Theory, 60(8), 5040-5053.
        https://arxiv.org/abs/1305.5870
        """
        u, s, v = linalg.svd(x, full_matrices=False)

        if trunc_svd == 0.:
            # hard threshold
            def _omega(_beta):
                return 0.56 * _beta ** 3 - 0.95 * _beta ** 2 + 1.82 * _beta + 1.43
            trunc_idx = jnp.sum(s > jnp.median(s) * _omega(jnp.divide(*sorted(x.shape))))
        elif (trunc_svd > 0.) & (trunc_svd < 1.):
            trunc_idx = jnp.sum(jnp.cumsum(s ** 2)/jnp.sum(s ** 2) < trunc_svd)
        elif (trunc_svd >= 1) & isinstance(trunc_svd, int):
            trunc_idx = trunc_svd
        else:
            trunc_idx = s.shape[0]

        return u[:, :trunc_idx], s[:trunc_idx], v.conj().T[:, :trunc_idx]

    @staticmethod
    def _tlsq(x0: jnp.ndarray, x1: jnp.ndarray, trunc_tlsq: int=0) -> tuple:
        """ Perform de-biasing DMD projection.
        :param x0: snapshot matrix as jax.numpy.ndarray
        :param x1: snapshot matrix as jax.numpy.ndarray
        :param trunc_tlsq: set tlsq truncation as int
        :return: Tuple containing projected snapshot matrices x0 and x1 as jax.numpy.ndarray.

        References:
        Matsumoto, D., & Indinger, T. (2017). On-the-fly algorithm for dynamic mode decomposition using incremental
        singular value decomposition and total least squares. arXiv preprint arXiv:1703.11004.
        https://arxiv.org/abs/1703.11004

        Hemati, M. S., Rowley, C. W., Deem, E. A., & Cattafesta, L. N. (2017). De-biasing the dynamic mode
        decomposition for applied Koopman spectral analysis of noisy datasets. Theoretical and Computational
        Fluid Dynamics, 31(4), 349-368.
        https://arxiv.org/pdf/1502.03854
        """
        if trunc_tlsq == 0:
            return x0, x1

        _, _, v = linalg.svd(jnp.append(x0, x1, axis=0), full_matrices=False)
        rank = min(v.shape[0], trunc_tlsq)
        vv = jnp.dot(v[:rank].conj().T, v[:rank])

        return jnp.dot(x0, vv), jnp.dot(x1, vv)

    @staticmethod
    def _vander(x: jnp.ndarray, n: int, pow: float=1) -> jnp.ndarray:
        """ Generate a Vandermonde matrix.
        :param x: jax.numpy.ndarray
        :param n: int describes the number of columns
        :param pow: float corresponding to the power of x per step
        :return: a Vandermond matrix  as jax.numpy.ndarray
        """
        return jnp.column_stack([x ** (i * pow) for i in range(n)])

    @staticmethod
    def _amplitude(x: jnp.ndarray, phi: jnp.ndarray) -> jnp.ndarray:
        """ Compute the amplitude of the dmd modes
        :param x: a vector as jax.numpy.ndarray
        :param phi: the dmd modes as jax.numpy.ndarray
        :return: The amplitudes as jax.numpy.ndarray
        """
        return linalg.lstsq(phi, x, rcond=None)[0]




