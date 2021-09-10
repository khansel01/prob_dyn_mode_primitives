import jax
import jax.numpy as jnp
import optax

from copy import deepcopy
from typing import Any, List, Optional, Tuple, Union, Dict
from tqdm import tqdm
from numpyro.distributions import MultivariateNormal
from models.parameter_classes import ParamClass, DistParamClass
from utils.utils_kernels import ard_kernel
from utils.utils_optim import gradient_loss, optimizer_step
from utils.utils_inference import posterior_u, posterior_as, posterior_y, posterior_0, posterior_x, posterior_a
from utils.utils_inference import smoothing_x
from utils.prng_handler import PRNGHandler
from utils.utils_math import inv, vander, pos_def


class GPDMD(object):
    def __init__(self, latent_dim: int, i_number: int, prng_handler: PRNGHandler, **kwargs):

        self.latent_dim = latent_dim

        self.prng_handler = prng_handler

        self.i_number = i_number
        i_vars = jnp.ones(shape=(self.i_number, self.latent_dim)) \
                 + jax.random.uniform(self.prng_handler.get_keys(1)[0],
                                      (self.i_number, self.latent_dim), minval=-1, maxval=1)

        self.kernel_fun = kwargs.get("kernel_fun", ard_kernel)
        gamma = kwargs.get("gamma", 1.)
        theta = kwargs.get("theta", 1.)
        self._opt_params = ParamClass({'x': None, "gamma": jnp.ones(latent_dim) * gamma, "theta": theta, "z": i_vars})

        lr_init = kwargs.get('lr_init', 1e-1)
        self.iterations_init = kwargs.get('iterations_init', 1000)
        self.optimizer_init = kwargs.get("optimizer", optax.adamw(learning_rate=lr_init))

        lr_main = kwargs.get('lr_main', None)
        self.iterations_main = kwargs.get('iterations_main', 500)
        if lr_main:
            self.optimizer_main = kwargs.get("optimizer", optax.adamw(learning_rate=lr_main))
        else:
            self.optimizer_main = None

        alpha_y = kwargs.get("alpha_y", 1. + 1e-16)
        beta_y = kwargs.get("beta_y", 1e-4)

        kappa_0 = kwargs.get("kappa_0", 1e-16)
        alpha_0 = kwargs.get("alpha_0", (1. + 1e-16))
        beta_0 = kwargs.get("beta_0",  1e-4)

        alpha_x = kwargs.get("alpha_x", 1. + 1e-16)
        beta_x = kwargs.get("beta_x", 1e-4)

        kappa_a = kwargs.get("kappa_a", 1e-16)
        alpha_a = kwargs.get("alpha_a", (1. + 1e-16))
        beta_a = kwargs.get("beta_a", 1e16)

        self._prior_params = DistParamClass({"alpha_y": alpha_y, "beta_y": beta_y,
                                            "alpha_0": jnp.ones(latent_dim) * alpha_0,
                                            "beta_0": jnp.ones(latent_dim) * beta_0,
                                            "alpha_x": alpha_x, "beta_x": beta_x,
                                            "alpha_a": jnp.ones((latent_dim, latent_dim)) * alpha_a,
                                            "beta_a": jnp.ones((latent_dim, latent_dim)) * beta_a,
                                            "kappa_0": kappa_0, "kappa_a": jnp.ones(latent_dim) * kappa_a})

        self._post_params = deepcopy(self._prior_params)
        lr_post = [1., 1e-1, 1., 1e-1, 1e-1, 1e-1]
        self.lr_post = kwargs.get("lr_post", lr_post)

        assert len(self.lr_post) == 6, ValueError(f'GPDMD contains of 6 closed form posterior updates. '
                                                  f'So the list lr_post should contain n=6 learning rates '
                                                  f'(given {len(self.lr_post)}).')

        self.ll_values = []

        self._trained_flag = False

    def __call__(self, data, *args, **kwargs):
        return self.fit(data)

    # ----------------------------------------------------------------------------------------------------------------
    # --------------------------------------------- property methods -------------------------------------------------
    # ----------------------------------------------------------------------------------------------------------------

    @property
    def post_params(self) -> DistParamClass:
        return self._post_params

    @post_params.setter
    def post_params(self, _post_params: DistParamClass) -> None:
        self._post_params.update(_post_params)

    @property
    def opt_params(self) -> ParamClass:
        return self._opt_params

    @opt_params.setter
    def opt_params(self, _opt_params: ParamClass) -> None:
        self._opt_params.update(_opt_params)

    @property
    def prior_params(self) -> DistParamClass:
        return self._prior_params

    @prior_params.setter
    def prior_params(self, _prior_params: DistParamClass) -> None:
        self._prior_params.update(_prior_params)

    @property
    def i_vars(self) -> jnp.ndarray:
        return self.opt_params.z

    @property
    def kernel_params(self) -> Tuple[jnp.ndarray, jnp.ndarray]:
        return self.opt_params.gamma, self.opt_params.theta

    @property
    def a_tilde(self) -> jnp.ndarray:
        return self.post_params.mu_a.T

    @property
    def mu_0(self) -> jnp.ndarray:
        return self.post_params.mu_0

    @property
    def _sigma_0(self) -> jnp.ndarray:
        return pos_def(jnp.diag(self.post_params.beta_0 / (self.post_params.alpha_0 * self.post_params.kappa_0)))

    @property
    def _lambda_0(self) -> jnp.ndarray:
        return pos_def(jnp.diag((self.post_params.alpha_0 * self.post_params.kappa_0) / self.post_params.beta_0))

    @property
    def mu_u(self) -> jnp.ndarray:
        return self.post_params.mu_u

    @property
    def mu(self) -> jnp.ndarray:
        return jnp.linalg.eigvals(self.a_tilde)

    @property
    def phi(self) -> jnp.ndarray:
        return jnp.linalg.eig(self.a_tilde)[1]

    @property
    def b(self) -> jnp.ndarray:
        return jnp.linalg.lstsq(self.phi, self.mu_0)[0]

    # ----------------------------------------------------------------------------------------------------------------
    # ----------------------------------------------- main functions -------------------------------------------------
    # ----------------------------------------------------------------------------------------------------------------

    def fit(self, data: jnp.ndarray) -> None:

        s, t, n = data.shape

        self.opt_params.update({'x': jnp.ones(shape=(s, t, self.latent_dim))})

        self.post_params.update({"mu_u": jnp.zeros((self.i_number, n)),
                                 "lambda_u": jnp.zeros((self.i_number, self.i_number)),
                                 "mu_as": jnp.zeros((s, self.latent_dim, self.latent_dim)),
                                 "lambda_as": jnp.zeros((s, self.latent_dim, self.latent_dim, self.latent_dim)),
                                 "mu_0": jnp.zeros(self.latent_dim),
                                 "mu_a": jnp.zeros((self.latent_dim, self.latent_dim))})

        optimizer = self.optimizer_init

        print(f'\t Pre-Training Phase')

        self._train(data, optimizer, self.lr_post, self.post_params, self.opt_params, self.prior_params,
                    self.kernel_fun, self.iterations_init, self.ll_values, main=False)

        print(f'\t Training Phase')

        if self.optimizer_main:
            optimizer = self.optimizer_main

        self._train(data, optimizer, self.lr_post, self.post_params, self.opt_params, self.prior_params,
                    self.kernel_fun, self.iterations_main, self.ll_values, main=True)

        self._trained_flag = True

        print(f'\t Finish')

        return None

    def predict(self, t_steps: jnp.ndarray, pow: Union[float] = 1.) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:

        assert self._trained_flag, ValueError(f'The GP-DMD has to be fitted before prediction is possible.')

        time_behaviour = vander(self.mu, len(t_steps), pow=pow)

        x = (self.phi @ jnp.diag(self.b) @ time_behaviour).T

        kernel_zz_inv = inv(
            self.kernel_fun((self.opt_params.gamma, self.opt_params.theta), self.i_vars.T, self.i_vars.T))

        kernel_tz = self.kernel_fun(self.kernel_params, x.T, self.i_vars.T)

        kernel_tt = self.kernel_fun(self.kernel_params, x.T, x.T)

        y = kernel_tz @ kernel_zz_inv @ self.mu_u

        cov = kernel_tt - kernel_tz @ kernel_zz_inv @ kernel_tz.T

        return x, y, cov

    def sampling(self, t_steps: jnp.ndarray, num_samples: Optional[int] = 1,
                 pow: Union[float] = 1.) -> Tuple[jnp.ndarray, jnp.ndarray]:

        assert self._trained_flag is not None, ValueError(f'The GP-DMD has to be fitted before sampling is possible.')

        time_behaviour = vander(self.mu, len(t_steps), pow=pow)

        kernel_zz_inv = inv(
            self.kernel_fun((self._opt_params.gamma, self._opt_params.theta), self.i_vars.T, self.i_vars.T))

        nrml_0 = MultivariateNormal(loc=jnp.real(self.mu_0), covariance_matrix=jnp.real(pos_def(self._sigma_0)))

        samples_0 = nrml_0.sample(self.prng_handler.get_keys(1)[0], sample_shape=(num_samples,))

        def _reconstruct_func(_: None, sample: List[jnp.ndarray]) -> Tuple:
            x_sample = (self.phi @ jnp.diag(jnp.linalg.lstsq(self.phi, sample[0], rcond=None)[0]) @ time_behaviour).T
            kernel_tz = self.kernel_fun(self.kernel_params, x_sample.T, self.i_vars.T)
            return None, [x_sample, kernel_tz @ kernel_zz_inv @ self.mu_u]

        x_samples, y_samples = jax.lax.scan(_reconstruct_func, None, xs=[samples_0])[1]

        return x_samples, y_samples

    # ----------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------ utility functions ---------------------------------------------
    # ----------------------------------------------------------------------------------------------------------------

    def as_dict(self) -> Dict:

        out_dict = {'i_number': self.i_number, 'latent_dim': self.latent_dim,
                    'post_params': dict(self.post_params), 'opt_params': dict(self.opt_params),
                    'prior_params': dict(self.prior_params), 'trained_flag': self._trained_flag}

        return out_dict

    def load_from_dict(self, _dict: Dict) -> None:

        self.prior_params = _dict['prior_params']

        self.post_params = _dict['post_params']

        self.opt_params = _dict['opt_params']

        self._trained_flag = _dict['trained_flag']

        return None

    # ----------------------------------------------------------------------------------------------------------------
    # ---------------------------------------------- static functions ------------------------------------------------
    # ----------------------------------------------------------------------------------------------------------------

    @staticmethod
    def _train(data: jnp.ndarray, optimizer: Any, lr_post: List, post_params: DistParamClass, opt_params: ParamClass,
               prior_params: DistParamClass, kernel_fun: Any, iterations: int, ll_values: List[float],
               main: Optional[bool] = True) -> jnp.ndarray:

        if main:
            argnums = (3, 4, 5, )
            opt_state = optimizer.init([opt_params.z, opt_params.gamma, opt_params.theta])
        else:
            argnums = (2, 3, 4, 5, )
            opt_state = optimizer.init([opt_params.x, opt_params.z, opt_params.gamma, opt_params.theta])

        for _iter in tqdm(range(iterations)):
            post_params.update(posterior_u(kernel_fun, post_params, lr_post[0], data, opt_params))

            post_params.update(posterior_y(kernel_fun, post_params, lr_post[1], data, opt_params, prior_params))

            if main:
                opt_params.x = smoothing_x(kernel_fun, post_params, data, opt_params)

            post_params.update(posterior_as(post_params, lr_post[2], opt_params))

            post_params.update(posterior_x(post_params, lr_post[3], opt_params, prior_params))

            post_params.update(posterior_0(post_params, lr_post[4], opt_params, prior_params))

            post_params.update(posterior_a(post_params, lr_post[5], prior_params))

            if main:
                params = [opt_params.z, opt_params.gamma, opt_params.theta]
            else:
                params = [opt_params.x, opt_params.z, opt_params.gamma, opt_params.theta]

            value, grads = gradient_loss(argnums, kernel_fun, data, opt_params, post_params)

            params, opt_state = optimizer_step(optimizer, params, opt_state, list(grads))

            if main:
                opt_params.z, opt_params.gamma, opt_params.theta = params
            else:
                opt_params.x, opt_params.z, opt_params.gamma, opt_params.theta = params

            ll_values.append(value)

        return opt_state


