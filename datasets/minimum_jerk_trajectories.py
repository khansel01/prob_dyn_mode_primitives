"""
This file generates Minimum-Jerk Trajectories.

References
Paper: A Minimum-Jerk Trajectory (http://courses.shadmehrlab.org/Shortcourse/minimumjerk.pdf)
Github: https://github.com/ekorudiawan/Minimum-Jerk-Trajectory
"""

from datasets.base_dataset import BaseDataset
import jax.numpy as jnp
import jax.random as random


class MinimumJerk(BaseDataset):
    def __init__(self, x_init: jnp.ndarray, x_final: jnp.ndarray, t_steps: jnp.ndarray, **kwargs):
        """  Generate batches of Minimum-Jerk trajectories.
        :param x_init: The initial x-values as a numpy nd.array of size Nx1.
        :param x_final: The final x-values as a numpy nd.array of size Nx1.
        :param t_steps: Time steps as a numpy nd.array of size T.
        :param kwargs:
                sigma: Describes the variability in initial and final x-values as a float value.
                s_size: Defines the sample size of the batch. It's specified as an int value.
        """
        super().__init__()
        self.x_init = x_init
        self.x_final = x_final
        self.t_steps = t_steps

        self.sigma = kwargs.get('sigma', 0.001)
        self.s_size = kwargs.get('s_size', 1)

        self.seed = kwargs.get('seed', 0)
        self.key = random.PRNGKey(self.seed)

    @property
    def transform(self) -> jnp.ndarray:
        """ Generate a batch of minimum Jerk trajectories
        :return: A batch of size b x n x m as numpy ndarray.
        B n and t correspond to the batch size the x_init size and the time, respectively.
        """
        x_init, x_final = self.x_random

        delta_x = x_final - x_init
        delta_t = self.t_steps / self.t_steps[-1]

        x = x_init + delta_x * (10 * delta_t ** 3 - 15 * delta_t ** 4 + 6 * delta_t ** 5)

        dx_dt = delta_x * (30 / self.t_steps[-1] * delta_t ** 2 - 60 / self.t_steps[-1] * delta_t ** 3
                           + 30 / self.t_steps[-1] * delta_t ** 4)

        dx_dtt = delta_x * (60 / self.t_steps[-1] ** 2 * delta_t - 180 / self.t_steps[-1] ** 2 * delta_t ** 2
                            + 120 / self.t_steps[-1] ** 2 * delta_t ** 3)

        return jnp.concatenate((x, dx_dt, dx_dtt), axis=1)

    @property
    def b_size(self) -> tuple:
        return self.s_size, self.x_init.shape[0], self.t_steps.shape[0]

    @property
    def x_random(self) -> tuple:
        key_init, key_final, self.key = random.split(self.key, 3)
        return random.normal(key_init, (*self.b_size[:2], 1)) * self.sigma + self.x_init, random.normal(
            key_final, (*self.b_size[:2], 1)) * self.sigma + self.x_final





