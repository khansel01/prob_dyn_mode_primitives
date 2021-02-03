""" The Python script Base Sampling DMD provides the default construct for all DMD scripts based on sampling
techniques like Gibbs Sampler.

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

from models.dmd_models.base_dmd import BaseDMD
from utilities.gibbs_sampler import GibbsSampler


class BaseSamplingDMD(BaseDMD):
    def __init__(self, **kwargs):
        """ Dynamic Mode Decomposition based on sampling techniques like Gibbs Sampler

        :param kwargs:
            alpha (float>0) : Initial shape of the gamma distribution.
            beta (float>0) : Initial rate of the gamma distribution.
            latent_dim (int>0) : Dimensionality of the latent space.
            prng_handler(class) : Class that handles the pseudo random number generation
            gibbs_iter(int) : Specifies the iterations of the Gibbs sampler
            gibbs_burn_in (int) : Specifies the Burn-in phase of the Gibbs sampler
        """
        super().__init__()
        # hyper parameters
        self.alpha = kwargs.get("alpha", 1e-3)
        self.beta = kwargs.get("beta", 1e-3)

        # Latent dimension
        self.latent_dim = kwargs.get("latent_dim", None)

        # PRNG Handler
        self.prng_handler = kwargs.get("prng_handler", None)

        # Initialize Gibbs_sampler
        conditionals = []

        self.sampler = GibbsSampler(conditionals,
                                    iterations=kwargs.get("gibbs_iter", None),
                                    burn_in=kwargs.get("gibbs_burn_in", None),
                                    prng_handler=kwargs.get("prng_handler", None))

    # override
    def fit(self, x0: jnp.ndarray, x1: jnp.ndarray, **kwargs) -> None:
        """ Compute the Dynamic Mode Decomposition given the two snapshot matrices x0 and x1

        :param x0: Snapshot matrix as jax numpy ndarray
        :param x1: Snapshot matrix as jax numpy ndarray
        """
        raise NotImplementedError(f"No fit() method available." 
                                  "BaseSampling DMD only specifies a default construct.")

    def _sampling(self, observations: list) -> None:
        """ Initializes the first sample and subsequently calls the Gibbs sampler to generate new samples.

        :param observations (list): Contains two jax devicearray corresponding to the two snapshot matrices
        :return: None
        """
        if self.latent_dim is None:
            raise ValueError('Dimensionality of latent space not specified. Please set latent_dim')
        if not self.prng_handler:
            raise ValueError(f"No PRNG Handler given.")

        # Initialize first sample
        keys = self.prng_handler.get_keys(3)
        sample_init = self._sample_init(keys, observations, self.latent_dim)
        args = (self.alpha, self.beta)

        # # Initialize Gibbs_sampler
        self.sampler.random_num = 2*self.latent_dim + 4
        self.sampler.sampling(observations, sample_init, *args)

    # -----------------------------------------------------------------------------------------------------------------
    # Next are methods associated with the initialization of the gibbs sampler
    # -----------------------------------------------------------------------------------------------------------------
    @staticmethod
    def _sample_init(keys, observations, latent_dim):
        """  This method initializes a sample given random keys, observations and the latent_dim

        :param keys (Jax device array): Pseudo random number generator (PRNG) Keys
        :param observations (list): Contains two jax devicearray corresponding to the two snapshot matrices
        :param latent_dim (int): Dimensionality of the latent space.
        :return:
        """
        raise NotImplementedError(f"No _sample_init() method available." 
                                  "BaseSampling DMD only specifies a default construct.")

    # -----------------------------------------------------------------------------------------------------------------
    # Next up are the getter and setter methods
    # -----------------------------------------------------------------------------------------------------------------
    @property
    def ll_values(self) -> list:
        """ Gets the list of all log-likelihood values.
        Each value represents the ll value of a sample from the Gibbs sampler sampling process..

        :return: List containing the log_likelihood values.
        """
        return self.sampler.ll_values

    @property
    def samples(self) -> list:
        """ Get a list containing all samples.

        :return: List of samples.
        """
        return list(map(list, zip(*self.sampler.samples)))

    @property
    def gibbs_iter(self) -> int:
        """ Get number of iterations of the gibbs sampler.

        :return: Integer
        """
        return self.sampler.iterations

    @gibbs_iter.setter
    def gibbs_iter(self, iterations: int) -> None:
        """ Set number of iterations of the gibbs sampler

        :param iterations (Int): Number of iterations.
        :return: None
        """
        self.sampler.iterations = iterations

    @property
    def gibbs_burn_in(self) -> int:
        """ Get the duration of the burn-in phase of the gibbs sampler.

        :return: Integer representing the duration of the burn-in phase.
        """
        return self.sampler.burn_in

    @gibbs_burn_in.setter
    def gibbs_burn_in(self, iterations: int) -> None:
        """ Set the duration of the burn-in phase of the gibbs sampler.

        :return: None
        """
        self.sampler.burn_in = iterations
