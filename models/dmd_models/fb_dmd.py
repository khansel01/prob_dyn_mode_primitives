""" An implementation of forward-backward dynamic mode decomposition based

Reference:
Dawson, S. T., Hemati, M. S., Williams, M. O., & Rowley, C. W. (2016). Characterizing and correcting for the effect of
sensor noise in the dynamic mode decomposition. Experiments in Fluids, 57(3), 42.

-----------------------------------------------------------------------------------------------------------------------
Several DMD methods in this project including forward-backward dDMD are inspired by the PyDMD package and have been
transferred to the jax library.

Reference:
Demo et al., (2018). PyDMD: Python Dynamic Mode Decomposition. Journal of Open Source Software, 3(22), 530
https://github.com/mathLab/PyDMD/tree/714ac2a9779890b16031d2c169e1eabdc46eeace

So if you're looking for a dynamic mode decomposition based on Python. Check out PyDMD!
-----------------------------------------------------------------------------------------------------------------------

TODO jax hasn't a matrix square root implementation. Scipy is used.
"""

import scipy as sp

import jax.numpy as jnp
from jax.numpy import linalg
from models.dmd_models.base_dmd import BaseDMD


class FBDMD(BaseDMD):
    def __init__(self):
        """ Forward-Backward Dynamic Mode Decomposition """
        super().__init__()

    # override
    def fit(self, x0: jnp.ndarray, x1: jnp.ndarray, **kwargs) -> None:
        """ Compute the Total-Least_squares Dynamic Mode Decomposition given the two snapshot matrices x0 and x1
        :param x0: Snapshot matrix as jax numpy ndarray
        :param x1: Snapshot matrix as jax numpy ndarray
        """
        trunc_svd = kwargs.get("trunc_svd", -1)
        trunc_tlsq = kwargs.get("trunc_tlsq", 0)

        # TLS truncation
        x0, x1 = self._tlsq(x0, x1, trunc_tlsq=trunc_tlsq)

        # SVD (foward and backward)
        u_r_f, s_r_f, v_r_f = self._svd(x0, trunc_svd=trunc_svd)
        u_r_b, s_r_b, v_r_b = self._svd(x1, trunc_svd=trunc_svd)

        if s_r_f.shape[0] != s_r_b.shape[0]:
            raise ValueError("Optimal truncation generates different numbers of singular values for the X and Y matrix,"
                             "please set different svd_rank.")

        # linear operators (foward and backward)
        a_forward = self._linear_op(x1, u_r_f, s_r_f, v_r_f)
        a_backward = self._linear_op(x0, u_r_b, s_r_b, v_r_b)

        # Compute lower dimensional linear operator
        self.a_tilde = sp.linalg.sqrtm(a_forward @ linalg.inv(a_backward))

        # Compute DMD Values and DMD Modes
        self.mu, self.phi = self._eig(self.a_tilde, u_r_f)

        # Compute Amplitude of DMD Modes
        self.b = self._amplitude(x0[:, 0], self.phi)

