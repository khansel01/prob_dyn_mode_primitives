""" Script implements a class that handles the pseudo random number generator (PRNG)."""

import jax.random as random


class PRNGHandler(object):
    def __init__(self, seed: int=0, **kwargs):
        """ Construct a class handling the pseudo random number generator.
        :param seed: Integer sets the seed
        :param kwargs: If given, prng_key is used instead of seed.
        """
        self.seed = seed
        self.key = kwargs.get("prng_key", random.PRNGKey(self.seed))

    def get_keys(self, number: int=1):
        """ Get a number of PRNG keys.
        :param number: Intiger specifies the number of PRNG keys
        :return: PRNG keys as Jax Devicearray
        """
        keys = random.split(self.key, number + 1)
        self.key = keys[-1]
        return keys[:-1]
