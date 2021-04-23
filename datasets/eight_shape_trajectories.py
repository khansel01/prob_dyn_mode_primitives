"""
This file generates eight shapes trajectories
"""

from datasets.base_dataset import BaseDataset
import jax.numpy as jnp
import jax.random as random
from jax.ops import index_update as i_update
from jax.ops import index as indx
import jax
from jax import jit


class EightShapes(BaseDataset):
    def __init__(self, t_steps: jnp.ndarray, **kwargs):
        """  Generate batches of eight shapes
        :param t_steps: Time steps as a numpy nd.array of size T.
        :param kwargs:
                sigma: Describes the variability of the eight shapes as a float value.
                s_size: Defines the sample size of the batch. It's specified as an int value.
                prng_handler: Class that handles the pseudo random number generator (PRNG)
        """
        super().__init__()
        self.t_steps = t_steps

        self.sigma = kwargs.get('sigma', 0.05)
        self.s_size = kwargs.get('s_size', 1)

        self.prng_handler = kwargs.get("prng_handler", None)

    @property
    def transform(self) -> jnp.ndarray:
        """ Generate a batch of eight shapes
        :return: A batch of size b x 3 x t as numpy ndarray.
        b and t correspond to the batch size and the time, respectively.
        """
        sigmas = self.random(self.prng_handler.get_keys(1))
        samples = jnp.zeros(self.b_size)

        return self._fun(samples, sigmas)

    @jax.partial(jit, static_argnums=(0,))
    def _fun(self, samples: jnp.ndarray, sigmas: jnp.ndarray) -> jnp.ndarray:
        for s in range(samples.shape[0]):
            samples = i_update(samples, indx[s, 0, :],
                               (sigmas[s, 0] + 2) * jnp.cos(2 * jnp.pi * self.t_steps + sigmas[s, 1]) + sigmas[s, 2])
            samples = i_update(samples, indx[s, 1, :],
                               (sigmas[s, 3] + 0.75) * jnp.sin(4 * jnp.pi * self.t_steps + sigmas[s, 4]) + sigmas[s, 5])
            samples = i_update(samples, indx[s, 2, :],
                               (sigmas[s, 6] + 0.75) * jnp.sin(4 * jnp.pi * self.t_steps + sigmas[s, 7]) + sigmas[s, 8])
        return samples

    @jax.partial(jit, static_argnums=(0,))
    def random(self, keys: tuple) -> jnp.ndarray:
        return random.normal(keys[0], shape=(*self.b_size[:1], 9)) * self.sigma

    # -----------------------------------------------------------------------------------------------------------------
    # Next up are the getter and setter methods
    # -----------------------------------------------------------------------------------------------------------------
    @property
    def b_size(self) -> tuple:
        return self.s_size, 3, self.t_steps.shape[0]

