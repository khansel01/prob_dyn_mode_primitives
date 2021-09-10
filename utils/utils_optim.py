import jax
import jax.numpy as jnp
import optax

from jax import jit, vmap
from utils.utils_math import inv
from models.parameter_classes import ParamClass, DistParamClass
from typing import Any, List, Tuple, Union, Sequence

# --------------------------------------------------------------------------------------------------------------------
# --------------------------------------- Loss And Optimization Functions --------------------------------------------
# --------------------------------------------------------------------------------------------------------------------


@jax.partial(jit, static_argnums=(0, ))
def _loss(kernel_fun: Any, y: jnp.ndarray, x: jnp.ndarray, z: jnp.ndarray, gamma: Union[float, jnp.ndarray],
         theta: float, post_params: DistParamClass) -> float:
    """ The Loss function of the GP-DMD used to optimize the point estimates of the kernel parameters, inducing inputs
    and depending on the algorithm the latent states.

    :param kernel_fun: A kernel function. Most of the time ARD Kernels are used.
    :param y: The data given as a jax.numpy.ndarray of shape S x T x N. S corresponds to the number of samples, T to
    the number of time steps, and N to the dimensions of the observation space.
    :param x: The latent states given as a jax.numpy.ndarray of shape S x T x M. S corresponds to the number of samples,
     T to the number of time steps, and M to the dimensions of the latent space.
    :param z: The inducing inputs given as a jax.numpy.ndarray of shape S x I x M. S corresponds to the number of
    samples, I to the number of inducing pairs, and M to the dimensions of the latent space.
    :param gamma: First kernel parameter which can be of type float or jax.numpy.ndarray.
    :param theta: Second kernel parameter which is of type float.
    :param post_params: Dict-like object containing all posterior parameters.
    :return:
    """

    k_zz_inv = inv(kernel_fun((gamma, theta), z.T, z.T))

    inv_lambda_u = inv(post_params.lambda_u)

    def f_loss(carry: float, inputs: List) -> Tuple:
        k_tz = kernel_fun((gamma, theta), inputs[0].T, z.T)
        k_tt = kernel_fun((gamma, theta), inputs[0].T, inputs[0].T)

        c_op = k_tz @ k_zz_inv
        d_op = k_tt - k_tz @ k_zz_inv @ k_tz.T

        _loss = 1 / 2 * (jnp.trace(jnp.diag(post_params.lambda_0)
                                   @ (post_params.mu_0[:, None].T @ post_params.mu_0[:, None]
                                      + post_params.kappa_0 * jnp.diag(post_params.lambda_0)))
                         + post_params.lambda_x
                         * jnp.trace((inputs[0][1:, :] - inputs[0][:-1, :] @ inputs[2]).T
                                     @ (inputs[0][1:, :] - inputs[0][:-1, :] @ inputs[2]))
                         + post_params.lambda_x
                         * jnp.trace(inputs[0][:-1, :] @ jnp.sum(vmap(inv)(inputs[3]), axis=0) @ inputs[0][:-1, :].T)
                         + post_params.lambda_y
                         * jnp.trace((inputs[1] - c_op @ post_params.mu_u) @ (inputs[1] - c_op @ post_params.mu_u).T)
                         + post_params.lambda_y * jnp.trace(y.shape[2] * c_op @ inv_lambda_u @ c_op.T)
                         + y.shape[2] * post_params.lambda_y * jnp.trace(d_op))
        return carry + _loss, None

    loss = 1 / 2 * (jnp.trace(k_zz_inv @ (post_params.mu_u @ post_params.mu_u.T + y.shape[2] * inv_lambda_u))
                    + y.shape[2] * jnp.linalg.slogdet(k_zz_inv)[1])

    return jax.lax.scan(f_loss, loss, xs=[x, y, post_params.mu_as, post_params.lambda_as])[0]


@jax.partial(jit, static_argnums=(0, 1, ))
def gradient_loss(argnums: Union[int, Sequence[int]], kernel_fun: Any, y: jnp.ndarray, opt_params: ParamClass,
                  post_params: DistParamClass) -> Tuple:
    """ Calculate the gradients and the log-loss of the Evidence Lower Bound.

    :param argnums:
    :param kernel_fun: A kernel function. Most of the time ARD Kernels are used.
    :param y: The data given as a jax.numpy.ndarray of shape S x T x N. S corresponds to the number of samples, T to
    the number of time steps, and N to the dimensions of the observation space.
    :param opt_params: Dict-like object containing all parameters which should be optimized.
    :param post_params: Dict-like object containing all posterior parameters.
    :return:
    """

    return jax.value_and_grad(_loss, argnums=argnums)(kernel_fun, y, opt_params.x, opt_params.z, opt_params.gamma,
                                                      opt_params.theta, post_params)


@jax.partial(jit, static_argnums=(0, ))
def optimizer_step(optimizer: Any, params: List, opt_state: List, gradients: List) -> Tuple:
    """ Take an optimization step of the parameters based on the given gradients.

    :param optimizer: Optimizer like ADAM or ADAMW.
    :param params: A list containing the parameters which should be optimized.
    :param opt_state: Jax opt_state of the optimizer.
    :param gradients: A List containing the gradients of the parameters.
    :return:
    """

    gradients, opt_state = optimizer.update(gradients, opt_state, params)

    return optax.apply_updates(params, gradients), opt_state
