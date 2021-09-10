import jax
import jax.numpy as jnp

from jax import jit, vmap, lax
from typing import Any, List, Dict, Tuple, Callable
from models.parameter_classes import ParamClass, DistParamClass
from utils.utils_math import inv, Gamma, Gaussian, GaussianGamma


# ---------------------------------------------------------------------------------------------------------------------
# ------------------------------------------ Inference Related Utility Functions --------------------------------------
# ---------------------------------------------------------------------------------------------------------------------


def filtering(data: jnp.ndarray, x_inital: jnp.ndarray, inital_sigma: jnp.ndarray, fx: jnp.ndarray,
              fx_sigma: jnp.ndarray, gx: Callable[[jnp.ndarray], jnp.ndarray], gx_sigma: jnp.ndarray) -> \
        Tuple[jnp.ndarray, jnp.ndarray]:
    """ A mixture of two popular Gaussian filtering algorithms is implemented, namely the Kalman filter and
    filtering based on spherical cubature integration. This combination is possible due to a Gaussian State-Space
    Model with nonlinear Emission Model and linear Transition Model. Filtering through the nonlinear part is
    achieved using approximations based on spherical cubature integration. For the linear dynamics, classical
    Kalman filtering takes place.

    :param data: is a TxN matrix. T and N correspond to the number of time steps and the number of dimensions in
    the observation space, respectively. This matrix represents the given data.
    :param x_inital: is a Mx1 matrix. M corresponds to the number of dimensions in the latent space. This matrix
    represents the initial state in the latent space.
    :param inital_sigma: is a MxM Matrix where M correspond to the number of dimensions in the latent space. This
    matrix represents a mostly diagonal covariance matrix of the initial latent state.
    :param fx: is a MxM Matrix where M correspond to the number of dimensions in the latent space. This matrix
    corresponds to the linear operator describing the linear dynamics in the latent space transitions.
    :param fx_sigma: is a MxḾ Matrix where M correspond to the number of dimensions in the latent space. This
    matrix represents a mostly diagonal covariance matrix and corresponds to the noise in the linear transition
    dynamics.
    :param gx: is a callable fxtion. This fxtion represents the nonlinear emission model and maps a current
    latent state into the observation space.
    :param gx_sigma: is a NxN Matrix where N correspond to the number of dimensions in the observation space. This
    matrix represents the mostly diagonal covariance matrix and corresponds to the noise in the emission model.
    :return: Tuple(mean, covariance):
            - mean: is a TxMx1 tensor. T and M correspond to the number of time steps and the number of dimensions,
            respectively. This tensor represents the estimated means of the filtering distribution denoted as ``mus``.
            - covariance: is a TxMxM tensor. T and N correspond to the number of time steps and the number of
            dimensions, respectively. This tensor represents the estimated covariances of the filtering distribution
            denoted as ``sigmas``.
    """

    n = fx.shape[0]

    unit_sigma_ps = jnp.concatenate((jnp.eye(n), - jnp.eye(n)), axis=1) * jnp.sqrt(n)

    def _forward_pass(init: Tuple[jnp.ndarray, jnp.ndarray], data: Tuple[jnp.ndarray, jnp.ndarray]) -> \
            Tuple[Tuple[jnp.ndarray, jnp.ndarray], Tuple[jnp.ndarray, jnp.ndarray]]:
        # initialize
        mu_old, sigma_old = init

        # prediction Step
        x_mu, x_sigma = jax.lax.cond(data[0] == 0, lambda _: (mu_old, sigma_old),
                                     lambda _: (fx @ mu_old, fx @ sigma_old @ fx.T + fx_sigma), operand=None)

        # update Step
        x_sigma_ps = x_mu + jnp.linalg.cholesky(x_sigma) @ unit_sigma_ps
        y_sigma_ps = gx(x_sigma_ps)
        y_mu = jnp.sum(y_sigma_ps, axis=1, keepdims=True) / (2 * n)
        y_sigma = (y_sigma_ps - y_mu) @ (y_sigma_ps - y_mu).T / (2 * n) + gx_sigma
        y_x_sigma = (x_sigma_ps - x_mu) @ (y_sigma_ps - y_mu).T / (2 * n)
        gain = y_x_sigma @ inv(y_sigma)

        mu = x_mu + gain @ (data[1][:, None] - y_mu)
        sigma = x_sigma - gain @ y_sigma @ gain.T
        return (mu, sigma), (mu, sigma)

    return jax.lax.scan(_forward_pass, (x_inital, inital_sigma), xs=(jnp.arange(data.shape[0]), data))[1]


def smoothing(f_means: jnp.ndarray, f_sigmas: jnp.ndarray, fx: jnp.ndarray, fx_sigma: jnp.ndarray) -> \
        Tuple[jnp.ndarray, jnp.ndarray]:
    """ Executing the Rauch—Tung—Striebel (RTS) smoother based on the previous results of a Bayesian filtering
    part.

    :param f_means: is a TxNx1 tensor. T and N correspond to the number of time steps and the number of dimensions,
    respectively. This tensor represents the estimated means denoted as mus during the filtering process.
    :param f_sigmas: is a TxNxN tensor. T and N correspond to the number of time steps and the number of dimensions,
    respectively. This tensor represents the estimated covariances denoted as sigmas during the filtering process.
    :param fx: is a NxN Matrix where N correspond to the number of dimensions. This matrix corresponds to the linear
    operator describing the linear dynamics in the transitions.
    :param fx_sigma: is a NxN Matrix where N correspond to the number of dimensions. This matrix represents a
    covariance matrix mostly diagonal and corresponds to the noise in the transition dynamics.
    :return: Tuple(mean, covariance):
            - mean: is a TxMx1 tensor. T and M correspond to the number of time steps and the number of dimensions,
            respectively. This tensor represents the estimated means of the smoothing distribution denoted as ``mus``.
            - covariance: is a TxMxM tensor. T and N correspond to the number of time steps and the number of
            dimensions, respectively. This tensor represents the estimated covariances of the smoothing distribution
            denoted as ``sigmas``.
    """

    def _backward_pass(init: Tuple[jnp.ndarray, jnp.ndarray], filtered_results: Tuple) -> \
            Tuple[Tuple[jnp.ndarray, jnp.ndarray], Tuple[jnp.ndarray, jnp.ndarray]]:
        # initialize
        mu_next, sigma_next = init
        mu_filtered, sigma_filtered = filtered_results

        # prediction Step
        _mu_next = fx @ mu_filtered
        _sigma_next = fx @ sigma_filtered @ fx.T + fx_sigma

        # update Step
        gain = sigma_next @ fx.T @ inv(_sigma_next)
        mu_smoothed = mu_filtered + gain @ (mu_next - _mu_next)
        sigma_smoothed = sigma_filtered + gain @ (sigma_next - _sigma_next) @ gain.T

        return (mu_smoothed, sigma_smoothed), (mu_smoothed, sigma_smoothed)

    return jax.lax.scan(_backward_pass, (jnp.zeros_like(f_means[-1]), jnp.zeros_like(f_sigmas[-1])),
                        xs=(f_means, f_sigmas), reverse=True)[1]

# --------------------------------------------------------------------------------------------------------------------
# -------------------- Smoothing Function To Sample From The Approximated Posterior Of The Latent Space --------------
# --------------------------------------------------------------------------------------------------------------------


@jax.partial(jit, static_argnums=(0, ))
def smoothing_x(kernel_fun: Any, post_params: DistParamClass, y: jnp.ndarray, opt_params: ParamClass) -> jnp.ndarray:
    """ Apply the combined Bayesian smoothing approach to get an mean estimate for each demonstration in the latent
    space. The combined Bayesian approach consists of a filtering and a smoothing algorithm. The former combines the
    classical Kalman Filtering for the linear transition model and Cubature integration for the nonlinear emission
    model. For the latter classical RTS smoothing is applied.

    :param kernel_fun: A kernel function. Most of the time ARD Kernels are used.
    :param post_params: Dict-like object containing all posterior parameters.
    :param y: The data given as a jax.numpy.ndarray of shape S x T x N. S corresponds to the number of samples, T to
    the number of time steps, and N to the dimensions of the observation space.
    :param opt_params: Dict-like object containing all optimization parameters.
    :return: Jax ndarray containing mean estimates for each latent space.
    """

    s, t, n = y.shape
    _, _, m = opt_params.x.shape

    k_zz_inv = inv(kernel_fun((opt_params.gamma, opt_params.theta), opt_params.z.T
                              , opt_params.z.T))

    def gx(x: jnp.ndarray) -> jnp.ndarray:
        def g(_: None, xs: jnp.ndarray) -> Tuple:
            return None, kernel_fun((opt_params.gamma, opt_params.theta), xs, opt_params.z.T) \
                   @ k_zz_inv @ post_params.mu_u

        out = vmap(lambda _x: jnp.concatenate(lax.scan(g, init=None, xs=_x.reshape(s, m, 1))[1],
                                              axis=1))(x[None, :, :].T)
        return jnp.concatenate(out, axis=0).T

    filter_results = filtering(data=jnp.concatenate(y, axis=1),
                               x_inital=jnp.kron(jnp.ones(s), post_params.mu_0).reshape(-1, 1),
                               inital_sigma=jnp.kron(jnp.eye(s), jnp.diag(jnp.reciprocal(post_params.lambda_0))
                                                     / post_params.kappa_0),
                               fx=jnp.kron(jnp.eye(s), post_params.mu_a.T),
                               fx_sigma=jnp.eye(s * m) / post_params.lambda_x, gx=gx,
                               gx_sigma=jnp.eye(s * n) / post_params.lambda_y)

    smoother_results = smoothing(f_means=filter_results[0], f_sigmas=filter_results[1],
                                 fx=jnp.kron(jnp.eye(s), post_params.mu_a.T),
                                 fx_sigma=jnp.eye(s * m) * post_params.beta_x / post_params.alpha_x)

    return smoother_results[0].transpose((2, 1, 0)).reshape(s, m, t).transpose((0, 2, 1))

# --------------------------------------------------------------------------------------------------------------------
# ------------------------------------------ Posterior Update Functions ----------------------------------------------
# --------------------------------------------------------------------------------------------------------------------


@jax.partial(jit, static_argnums=(0, ))
def posterior_u(kernel_fun: Any, post_params: DistParamClass, l_r: float, y: jnp.ndarray,
                opt_params: ParamClass) -> Dict:
    """ Determines the parameters of Gaussian posteriors belonging to the inducing variables of the emission model. If a
    learning rate smaller than 1 is chosen, an update of the previous posterior parameters towards the new parameters
    in the natural parameter space takes place.

    :param kernel_fun: A kernel function. Most of the time ARD Kernels are used.
    :param post_params: Dict-like object containing all posterior parameters.
    :param l_r: A float value representing the learning rate.
    :param y: The data given as a jax.numpy.ndarray of shape S x T x N. S corresponds to the number of samples, T to
    the number of time steps, and N to the dimensions of the observation space.
    :param opt_params: Dict-like object containing all optimization parameters.
    :return: Dictionary containing the the parameters of a Gaussian posterior.
    """

    k_zz_inv = inv(kernel_fun((opt_params.gamma, opt_params.theta), opt_params.z.T, opt_params.z.T))

    def f_scan(carry: List, inputs: List) -> Tuple:
        k_tz = kernel_fun((opt_params.gamma, opt_params.theta), inputs[0].T, opt_params.z.T)
        return [carry[0] + k_tz.T @ inputs[1], carry[1] + k_tz.T @ k_tz], None

    psi_3, psi_4 = jax.lax.scan(f_scan, [jnp.zeros((opt_params.z.shape[0], y.shape[2])),
                                         jnp.zeros((opt_params.z.shape[0], opt_params.z.shape[0]))],
                                xs=[opt_params.x, y])[0]

    eta_1, eta_2 = Gaussian.std_to_nat([post_params.mu_u, post_params.lambda_u])

    _mu, _lambda = Gaussian.nat_to_std([(1 - l_r) * eta_1 + l_r * (post_params.lambda_y * k_zz_inv @ psi_3), (
            1 - l_r) * eta_2 - l_r / 2 * (k_zz_inv + post_params.lambda_y * k_zz_inv @ psi_4 @ k_zz_inv)])

    return {"mu_u": _mu, "lambda_u": _lambda}


@jit
def posterior_as(post_params: DistParamClass, l_r: float, opt_params: ParamClass) -> Dict:
    """ Determines the parameters of Gaussian posteriors belonging to the linear operators of each demonstration. If a
    learning rate smaller than 1 is chosen, an update of the previous posterior parameters towards the new parameters
    in the natural parameter space takes place.

    :param post_params: Dict-like object containing all posterior parameters.
    :param l_r: Dict-like object containing all posterior parameters.
    :param opt_params: Dict-like object containing all optimization parameters.
    :return: Dictionary containing the the parameters of a Gaussian posterior.
    """

    def f_vmap(_p_0: jnp.ndarray, _p_1: jnp.ndarray, _x: jnp.ndarray) -> List:
        psi_1 = post_params.lambda_x * _x[:-1].T @ _x[1:]
        psi_2 = post_params.lambda_x * _x[:-1].T @ _x[:-1]
        return jax.lax.scan(f_scan, psi_2, xs=[_p_0.T, _p_1, post_params.mu_a.T, post_params.lambda_a, psi_1.T])[1]

    def f_scan(carry: jnp.ndarray, inputs: List) -> Tuple:
        eta_1, eta_2 = Gaussian.std_to_nat([inputs[0][:, None], inputs[1]])
        _mu, _lambda = Gaussian.nat_to_std([(1 - l_r) * eta_1 + l_r * (inputs[4][:, None] + jnp.diag(inputs[3])
                                                                       @ inputs[2][:, None]),
                                            (1 - l_r) * eta_2 - l_r / 2 * (jnp.diag(inputs[3]) + carry)])
        return carry, [_mu[:, 0], _lambda]

    _mus, _lambdas = vmap(f_vmap)(post_params.mu_as, post_params.lambda_as, opt_params.x)

    return {"mu_as": jnp.transpose(_mus, (0, 2, 1)), "lambda_as": _lambdas}


@jax.partial(jit, static_argnums=(0, ))
def posterior_y(kernel_fun: Any, post_params: DistParamClass, l_r: float, y:jnp.ndarray, opt_params: ParamClass,
                prior_params: DistParamClass) -> Dict:
    """ Determines the parameters of a Gamma posterior belonging to the precision of the emission model. If a learning
    rate smaller than 1 is chosen, an update of the previous posterior parameters towards the new parameters in the
    natural parameter space takes place.

    :param kernel_fun: A kernel function. Most of the time ARD Kernels are used.
    :param post_params: Dict-like object containing all posterior parameters.
    :param l_r: Dict-like object containing all posterior parameters.
    :param y: The data given as a jax.numpy.ndarray of shape S x T x N. S corresponds to the number of samples, T to
    the number of time steps, and N to the dimensions of the observation space.
    :param opt_params: Dict-like object containing all optimization parameters.
    :param prior_params: Dict-like object containing all prior parameters.
    :return: Dictionary containing the the parameters of a Gamma posterior.
    """

    k_zz_inv = inv(kernel_fun((opt_params.gamma, opt_params.theta), opt_params.z.T, opt_params.z.T))

    def f_trace(carry: float, inputs: List) -> Tuple:
        k_tz = kernel_fun((opt_params.gamma, opt_params.theta), inputs[0].T, opt_params.z.T)
        k_tt = kernel_fun((opt_params.gamma, opt_params.theta), inputs[0].T, inputs[0].T)

        c_op = k_tz @ k_zz_inv
        d_op = k_tt - k_tz @ k_zz_inv @ k_tz.T

        _trace = jnp.trace((inputs[1] - c_op @ post_params.mu_u) @ (inputs[1]- c_op @ post_params.mu_u).T) \
                 + y.shape[2] * jnp.trace(d_op + c_op @ inv(post_params.lambda_u) @ c_op.T)
        return carry + _trace, None

    trace = jax.lax.scan(f_trace, 0, xs=[opt_params.x, y])[0]

    eta_1, eta_2 = Gamma.std_to_nat([post_params.alpha_y, post_params.beta_y])

    _alpha, _beta = Gamma.nat_to_std([(1 - l_r) * eta_1 + l_r * (prior_params.alpha_y + y.size / 2 - 1),
                                      (1 - l_r) * eta_2 - l_r * (prior_params.beta_y + 1 / 2 * trace)])

    return {"alpha_y": _alpha, "beta_y": _beta}


@jit
def posterior_0(post_params: DistParamClass, l_r: float, opt_params: ParamClass,
                prior_params: DistParamClass) -> Dict:
    """ Determines the parameters of a spherical Gaussian Wishart posterior belonging to the hierarchical model of the
    initial state in the latent space. If a learning rate smaller than 1 is chosen, an update of the previous posterior
    parameters towards the new parameters in the natural parameter space takes place.

    :param post_params: Dict-like object containing all posterior parameters.
    :param l_r: Dict-like object containing all posterior parameters.
    :param opt_params: Dict-like object containing all optimization parameters.
    :param prior_params: Dict-like object containing all prior parameters.
    :return: Dictionary containing the the parameters of a spherical Gaussian Wishart posterior.
    """

    mean_x_0 = jnp.sum(opt_params.x[:, 0, :, None], axis=0) / opt_params.x.shape[0]

    _kappa = prior_params.kappa_0 + opt_params.x.shape[0]

    _mu = opt_params.x.shape[0] * mean_x_0 / _kappa

    _alpha = prior_params.alpha_0[:, None] + opt_params.x.shape[0] / 2

    _beta = (opt_params.x[:, 0].T - mean_x_0) @ (opt_params.x[:, 0].T - mean_x_0).T \
            + prior_params.kappa_0 * opt_params.x.shape[0] * mean_x_0 @ mean_x_0.T / _kappa

    _beta = prior_params.beta_0[:, None] + 1 / 2 * jnp.diag(_beta)[:, None]

    eta_1, eta_2, eta_3, eta_4 = GaussianGamma.std_to_nat([post_params.mu_0[:, None], post_params.kappa_0,
                                                           post_params.alpha_0[:, None], post_params.beta_0[:, None]])

    _mu, _kappa, _alpha, _beta = GaussianGamma.nat_to_std([
        (1 - l_r) * eta_1 + l_r * (_kappa * _mu),
        (1 - l_r) * eta_2 + l_r * _kappa,
        (1 - l_r) * eta_3 + l_r * (2 * _alpha - 1),
        (1 - l_r) * eta_4 + l_r * (2 * _beta + _kappa * _mu ** 2)
    ])

    return {"mu_0": _mu[:, 0], "kappa_0": _kappa, "alpha_0": _alpha[:, 0], "beta_0": _beta[:, 0]}


@jit
def posterior_x(post_params: DistParamClass, l_r: float, opt_params: ParamClass,
                prior_params: DistParamClass) -> Dict:
    """ Determines the parameters of a Gamma posterior belonging to the precision of the transition model. If a learning
    rate smaller than 1 is chosen, an update of the previous posterior parameters towards the new parameters in the
    natural parameter space takes place.


    :param post_params: Dict-like object containing all posterior parameters.
    :param l_r: Dict-like object containing all posterior parameters.
    :param opt_params: Dict-like object containing all optimization parameters.
    :param prior_params: Dict-like object containing all prior parameters.
    :return: Dictionary containing the the parameters of a Gamma posterior.
    """

    def f_trace(carry, inputs):
        _trace = jnp.trace((inputs[0][1:, :] - inputs[0][:-1, :] @ inputs[1]).T @
                           (inputs[0][1:, :] - inputs[0][:-1, :] @ inputs[1])) \
                 + jnp.trace(inputs[0][:-1, :].T @ inputs[0][:-1, :] @ jnp.sum(vmap(inv)(inputs[2]), axis=0))
        return carry + _trace, None

    trace = jax.lax.scan(f_trace, 0, xs=[opt_params.x, post_params.mu_as, post_params.lambda_as])[0]

    eta_1, eta_2 = Gamma.std_to_nat([post_params.alpha_x, post_params.beta_x])

    _alpha, _beta = Gamma.nat_to_std([(1 - l_r) * eta_1
                                      + l_r * (prior_params.alpha_x + (opt_params.x.shape[0]
                                                                       * (opt_params.x.shape[1] - 1)
                                                                       * opt_params.x.shape[2] + 1) / 2 - 1),
                                      (1 - l_r) * eta_2 - l_r * (prior_params.beta_x + 1 / 2 * trace)])

    return {"alpha_x": _alpha, "beta_x": _beta}


@jit
def posterior_a(post_params: DistParamClass, l_r: float, prior_params: DistParamClass) -> Dict:
    """ Determines the parameters of a spherical Gaussian Wishart posterior belonging to the hierarchical model of the
    linear operator. If a learning rate smaller than 1 is chosen, an update of the previous posterior parameters towards
    the new parameters in the natural parameter space takes place.

    :param post_params: Dict-like object containing all posterior parameters.
    :param l_r: Dict-like object containing all posterior parameters.
    :param prior_params: Dict-like object containing all prior parameters.
    :return: Dictionary containing the the parameters of a spherical Gaussian Wishart posterior.
    """

    def f_scan(carry: jnp.ndarray, inputs: List) -> Tuple:
        mean_mu = jnp.sum(inputs[3][:, :, None], axis=0) / carry
        _kappa = inputs[0] + carry
        _mu = carry * mean_mu / _kappa
        _alpha = inputs[1][:, None] + carry / 2
        _beta = (inputs[3].T - mean_mu) @ (inputs[3].T - mean_mu).T + jnp.sum(vmap(inv)(inputs[4]), axis=0) \
                + inputs[0] * carry * mean_mu @ mean_mu.T / _kappa
        _beta = inputs[2][:, None] + 1 / 2 * jnp.diag(_beta)[:, None]

        eta_1, eta_2, eta_3, eta_4 = GaussianGamma.std_to_nat([inputs[5][:, None], inputs[6], inputs[7][:, None],
                                                               inputs[8][:, None]])
        _mu, _kappa, _alpha, _beta = GaussianGamma.nat_to_std([
            (1 - l_r) * eta_1 + l_r * (_kappa * _mu),
            (1 - l_r) * eta_2 + l_r * _kappa,
            (1 - l_r) * eta_3 + l_r * (2 * _alpha - 1),
            (1 - l_r) * eta_4 + l_r * (2 * _beta + _kappa * _mu ** 2)
        ])
        return carry, [_mu[:, 0], _kappa, _alpha[:, 0], _beta[:, 0]]

    mu, kappa, alpha, beta = jax.lax.scan(f_scan, post_params.mu_as.shape[0],
                                          xs=[prior_params.kappa_a, prior_params.alpha_a, prior_params.beta_a,
                                              post_params.mu_as.transpose((2, 0, 1)),
                                              post_params.lambda_as.transpose((1, 0, 2, 3)),
                                              post_params.mu_a.T, post_params.kappa_a,
                                              post_params.alpha_a, post_params.beta_a])[1]

    return {"mu_a": mu.T, "kappa_a": kappa, "alpha_a": alpha, "beta_a": beta}

