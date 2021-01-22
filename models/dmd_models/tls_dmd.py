""" An implementation of total-least-squares dynamic mode decomposition based

Reference:
Dawson, S. T., Hemati, M. S., Williams, M. O., & Rowley, C. W. (2016). Characterizing and correcting for the effect of
sensor noise in the dynamic mode decomposition. Experiments in Fluids, 57(3), 42.

-----------------------------------------------------------------------------------------------------------------------
Several DMD methods in this project are inspired by the PyDMD package and have been
transferred to the jax library.

Reference:
Demo et al., (2018). PyDMD: Python Dynamic Mode Decomposition. Journal of Open Source Software, 3(22), 530
https://github.com/mathLab/PyDMD/tree/714ac2a9779890b16031d2c169e1eabdc46eeace

So if you're looking for a dynamic mode decomposition based on Python. Check out PyDMD!
-----------------------------------------------------------------------------------------------------------------------
"""

import jax.numpy as jnp
from models.dmd_models.base_dmd import BaseDMD


class TLSDMD(BaseDMD):
    def __init__(self):
        """ Total-Least-Squares Dynamic Mode Decomposition """
        super().__init__()

    # override
    def fit(self, x0: jnp.ndarray, x1: jnp.ndarray, **kwargs) -> None:
        """ Compute the Total-Least_squares Dynamic Mode Decomposition given the two snapshot matrices x0 and x1
        :param x0: Snapshot matrix as jax numpy ndarray
        :param x1: Snapshot matrix as jax numpy ndarray
        """
        trunc_svd = kwargs.get('trunc_svd', x0.shape[0]//2)

        # Collect data and project them onto r POD modes
        data = jnp.append(x0, x1[:, -1:], axis=1)
        u_r, s_r, v_r = self._svd(data, trunc_svd=trunc_svd)
        _data = u_r.T @ data

        # Compute lower dimensional linear operator
        self.a_tilde = self._linear_op(_data[:, :-1], _data[:, 1:])

        # Compute DMD Values and DMD Modes
        self.mu, self.phi = self._eig(self.a_tilde, x1, s_r, v_r[:-1])

        # Compute Amplitude of DMD Modes
        self.b = self._amplitude(x0[:, 0], self.phi)
