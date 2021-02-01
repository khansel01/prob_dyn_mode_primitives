""" An implementation of exact dynamic mode decomposition based.

Reference:
Tu, Jonathan H., et al. "On dynamic mode decomposition: Theory and applications." arXiv preprint arXiv:1312.0041 (2013).
https://arxiv.org/abs/1312.0041

-----------------------------------------------------------------------------------------------------------------------
Most DMD methods in this project including ExactDMD are inspired by the PyDMD package and have been transferred
to the jax library.

Reference:
Demo et al., (2018). PyDMD: Python Dynamic Mode Decomposition. Journal of Open Source Software, 3(22), 530
https://github.com/mathLab/PyDMD/tree/714ac2a9779890b16031d2c169e1eabdc46eeace

So if you're looking for a dynamic mode decomposition based on Python. Check out PyDMD!
-----------------------------------------------------------------------------------------------------------------------
"""

import jax.numpy as jnp
from models.dmd_models.base_dmd import BaseDMD


class ExactDMD(BaseDMD):
    def __init__(self, **kwargs):
        """ Exact Dynamic Mode Decomposition  """
        super().__init__()

    # override
    def fit(self, x0: jnp.ndarray, x1: jnp.ndarray, **kwargs) -> None:
        """ Compute the Total-Least_squares Dynamic Mode Decomposition given the two snapshot matrices x0 and x1
        :param x0: Snapshot matrix as jax numpy ndarray
        :param x1: Snapshot matrix as jax numpy ndarray
        """
        trunc_svd = kwargs.get("trunc_svd", 0)
        trunc_tls = kwargs.get("trunc_tls", 0)

        # TLS truncation
        x0, x1 = self._tlsq(x0, x1, trunc_tlsq=trunc_tls)

        # SVD
        u_r, s_r, v_r = self._svd(x0, trunc_svd=trunc_svd)

        # Compute lower dimensional linear operator
        self.a_tilde = self._linear_op(x1, u_r, s_r, v_r)

        # Compute DMD Values and DMD Modes
        self.mu, self.phi = self._eig(self.a_tilde, x1, s_r, v_r)

        # Compute Amplitude of DMD Modes
        self.b = self._amplitude(x0[:, 0], self.phi)

