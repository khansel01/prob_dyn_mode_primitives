""" Gibbs Sampler """

from tqdm import tqdm


class GibbsSampler(object):
    def __init__(self, conditionals: list, **kwargs):
        """ Gibbs sampler

        :param conditionals (list): Contains conditional distribution functions
        :param kwargs:
            random_num (int>0) : Number of random values.
            prng_handler(class) : Class that handles the pseudo random number generation
            gibbs_iter(int) : Specifies the iterations of the Gibbs sampler
            gibbs_burn_in (int) : Specifies the Burn-in phase of the Gibbs sampler
            ll_fun (function) : Log likelihood function
        """
        # List containing conditional functions
        self.conditionals = conditionals

        # Number of Random Variables
        self.random_num = kwargs.get("random_num", len(conditionals))

        # Number of iterations and burn in phase of the gibbs sampler
        self.iterations = kwargs.get("iterations", 1000)
        self.burn_in = kwargs.get("burn_in", 500)

        # PRNG key & seed
        self.prng_handler = kwargs.get("prng_handler", None)

        # Log Likelihood fct and value
        self.ll_fun = kwargs.get("ll_fun", None)
        self.ll_values = []

        # Samples
        self.samples = []

    def sampling(self, observations: list, sample_init: list, *args) -> None:
        """ Start sampling process.

        :param observations: List of two jax devicearray corresponding to the two snapshot matrices.
        :param sample_init: List containing the initial sample..
        :param args: Possible arguments for the conditional functions.
        :return: None
        """
        if not self.conditionals:
            raise ValueError(f"No conditional functions were initialized.")
        if not self.prng_handler:
            raise ValueError(f"No PRNG Handler given.")

        sample = sample_init

        for g_iter in tqdm(range(self.iterations)):
            keys = self.prng_handler.get_keys(self._random_num)

            for conditional in self.conditionals:
                keys, sample = conditional(keys, observations, sample, *args)

            if self.ll_fun:
                self.ll_values.append(self.ll_fun(observations, sample))

            if g_iter >= self.burn_in:
                self.samples.append(sample)

    @property
    def random_num(self) -> int:
        """ Get the Number of random variables.

        :return: Number of random variables as integer.
        """
        return self._random_num

    @random_num.setter
    def random_num(self, random_num: int) -> None:
        """ Set the number of random variables.

        :param random_num: Number of random variables as integer.
        :return: None.
        """
        self._random_num = random_num

