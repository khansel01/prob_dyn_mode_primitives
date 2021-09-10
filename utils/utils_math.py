import jax
import jax.numpy as jnp

from jax import jit, lax
from typing import Any
from jax.scipy.special import gammaln
from numpyro.distributions import MultivariateNormal
from typing import List, Tuple

# ---------------------------------------------------------------------------------------------------------------------
# -------------------------------------------- Math Related Utility Functions -----------------------------------------
# ---------------------------------------------------------------------------------------------------------------------


@jit
def inv(x: jnp.ndarray) -> jnp.ndarray:
    """ Calculate the inverse of an N x N matrix based on least squares.

    :param x: an N x N matrix as jnp.ndarray
    :return: the inverse of x as N x N matrix as jnp.ndarray
    """

    return jnp.linalg.lstsq(x, jnp.eye(x.shape[0]), rcond=-1)[0]


@jit
def pos_def(x: jnp.ndarray) -> jnp.ndarray:
    """ Calculate closest positive-definite symmetric NxN Matrix.

    :param x: Input matrix as N x N Matrix of type jnp.ndarray.
    :return: Closest sym. positive-definite N x N Matrix of type jnp.ndarray.
    """

    def closest_matrix(b: jnp.ndarray) -> jnp.ndarray:
        out = (b + b.conj().T) / 2
        eig_val, eig_vec = jnp.linalg.eig(out)
        out = eig_vec @ jnp.diag(jnp.maximum(eig_val, 1e-5)) @ eig_vec.conj().T
        return out.astype(x.dtype)

    return lax.cond(jnp.all(jnp.linalg.eigvals(x) > 0) & jnp.all(jnp.isclose(x, x.conj().T)),
                    lambda b: b.astype(x.dtype), closest_matrix, x)


@jax.partial(jit, static_argnums=(1,))
def vander(x: jnp.ndarray, n: int, pow: float = 1.) -> jnp.ndarray:
    """ Generate a Vandermond matrix.

    :param x: input array of type jnp.ndarray.
    :param n: The number of columns of the output matrix.
    :param pow: optional floating parameter.
    :return: A Vandermond matrix as jnp.ndarray.
    """

    def f_scan(_: None, i: int) -> Tuple:
        return None, x ** (i * pow)

    return lax.scan(f_scan, None, xs=jnp.arange(n))[1].T


@jit
def sample_complex_normal(key: Any, mean: jnp.ndarray, sigma: jnp.ndarray):
    """ Sampling from a complex multivariate normal distribution

    :param key: Pseudo random number generator (PRNG) Keys
    :param mean: as a complex jax device array
    :param sigma: positive definite symmetric covariance matrix
    :return:
    """

    if jnp.ndim(mean) == 1:
        mean = mean[:, None]

    dim = mean.shape[0]

    _mean = jnp.append(jnp.real(mean), jnp.imag(mean), axis=0)

    _sigma = 0.5 * jnp.vstack((jnp.hstack((jnp.real(sigma), -jnp.imag(sigma))),
                               jnp.hstack((jnp.imag(sigma), jnp.real(sigma)))))

    _sigma = 0.5 * (_sigma + _sigma.conj().T)

    _normal = MultivariateNormal(loc=_mean.T, covariance_matrix=_sigma)

    samples = _normal.sample(key)

    return samples[:, :dim] + 1j * samples[:, dim:]

# ---------------------------------------------------------------------------------------------------------------------
# ------------------------------------------ Math Related Utility Classes ---------------------------------------------
# ---------------------------------------------------------------------------------------------------------------------


class Gamma(object):
    """ Contains the functions of a Gamma distribution to convert standard parameters to natural parameters and
    vice versa.

    """

    @staticmethod
    @jit
    def std_to_nat(std_param: List) -> List:
        """ Transform the standard parameters of a Gamma distribution into the natural parameters.

        :param std_param: A list containing the shape and the rate parameters of a Gamma distribution.
        :return: The natural parameters as list.
        """

        nat_param1 = std_param[0] - 1

        nat_param2 = - std_param[1]

        return [nat_param1, nat_param2]

    @staticmethod
    @jit
    def nat_to_std(nat_param: List) -> List:
        """ Transform the natural parameters of a Gamma distribution in exponential family form back to the standard
        shape and rate parameters.

        :param nat_param: The natural parameters of type list.
        :return: A list containing the shape and the rate of a Gamma distribution.
        """

        alpha = nat_param[0] + 1

        beta = - nat_param[1]

        return [alpha, beta]

    @staticmethod
    @jit
    def log_likelihood(std_param: List, x: jnp.ndarray) -> float:
        """ Calculate the log likelihood of a Gamma distribution.

        :param std_param: A list containing the shape and the rate parameters of a Gamma distribution.
        :param x: The given data as jax.numpy.ndarray.
        :return: A float representing the log likelihood
        """

        ll_value = (std_param[0] - 1.) * jnp.log(x) - std_param[1] * x

        ll_value -= gammaln(std_param[0]) - std_param[0] * jnp.log(std_param[1])

        return ll_value


class Gaussian(object):
    """ Contains the functions of a Gaussian distribution to convert standard parameters to natural parameters and
    vice versa.

    """

    @staticmethod
    @jit
    def std_to_nat(std_param: List) -> List:
        """ Transform the standard parameters of a multivariate Gaussian distribution into the natural parameters.

        :param std_param: A list containing the mean and the precision parameters of a Gaussian distribution.
        :return: The natural parameters as list.
        """

        nat_param1 = std_param[1] @ std_param[0]

        nat_param2 = - 1/2 * std_param[1]

        return [nat_param1, nat_param2]

    @staticmethod
    @jit
    def nat_to_std(nat_param: List) -> List:
        """ Transform the natural parameters of a multivariate Gaussian distribution in exponential form back to
        the mean and precision parameters of the standard form.

        :param nat_param: The natural parameters of type list.
        :return: A list containing the mean and the precision of a Gamma distribution.
        """

        mu = - 1/2 * inv(nat_param[1]) @ nat_param[0]

        lamda = - 2 * nat_param[1]

        return [mu, lamda]

    @staticmethod
    @jit
    def log_likelihood(std_param: List, x: jnp.ndarray) -> float:
        """ Calculate the log likelihood of a Gaussian distribution.

        :param std_param: A list containing the shape and the rate parameters of a Gaussian distribution.
        :param x: The given data as jax.numpy.ndarray.
        :return: A float representing the log likelihood
        """

        ll_value = jnp.einsum('ml, mi, ik -> ', std_param[0], std_param[1], x) \
                   - 1/2 * jnp.einsum('mk, mi, ik -> ', x, std_param[1], x)

        ll_value -= x.shape[1]/2 * jnp.einsum('mk, mi, ik -> ', std_param[0], std_param[1], std_param[0]) \
                    - jnp.sum(jnp.log(jnp.diag(jnp.linalg.cholesky(std_param[1]))))

        return ll_value - x.shape[1]/2 * jnp.log(2 * jnp.pi)


class GaussianGamma(object):
    """ Contains the functions of a Gaussian Gamma distribution to convert standard parameters to natural parameters
    and vice versa.

    """

    @staticmethod
    @jit
    def std_to_nat(std_param: List) -> List:
        """ Transform the standard parameters of a Gaussian Gamma distribution into the natural parameters.

        :param std_param: A list containing the mean, the precision, the shape, and the rate parameters of a
        Gaussian Gamma distribution.
        :return: The natural parameters as list.
        """

        a = std_param[1] * std_param[0]

        b = std_param[1]

        c = 2. * std_param[2] - 1.

        d = 2. * std_param[3] + std_param[1] * std_param[0]**2

        return [a, b, c, d]

    @staticmethod
    @jit
    def nat_to_std(nat_param: List) -> List:
        """ Transform the natural parameters of a Gaussian Gamma distribution in exponential form back to
        the mean, the precision, the shape, and the rate parameters of the standard form.

        :param nat_param:
        :return:
        """

        mu = nat_param[0] / nat_param[1]

        kappas = nat_param[1]

        alphas = 0.5 * (nat_param[2] + 1.)

        betas = 0.5 * (nat_param[3] - kappas * mu**2)

        return [mu, kappas, alphas, betas]


