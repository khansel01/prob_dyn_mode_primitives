import matplotlib.pyplot as plt
import argparse

from datasets import MinimumJerk
from utilities.prng_handler import PRNGHandler
from utilities.kernel_lib import *
from utilities.utils import *
from tqdm import tqdm
from numpyro.distributions import MultivariateNormal
from jax.ops import index, index_update

parser = argparse.ArgumentParser(description="Test")
parser.add_argument("--device", default="cpu", type=str,
                    help="Specify the device on which Jax works. Possible devices are cpu or gpu (if available).")
parser.add_argument("--x64", default=True, type=bool,
                    help="Defines the float type used by Jax.")

parser.add_argument("--seed", default=11, type=int,
                    help="Defines the seed of the pseudo random number generator.")
parser.add_argument("--sigma", default=0.1, type=float,
                    help="Set the variability in the sampled data.")

parser.add_argument("--latent_dim", default=5, type=int,
                    help="Dimensionality of the latent space.")
parser.add_argument("--iterations", default=5000, type=int,
                    help="Specifies the iterations of the Optimizer.")
parser.add_argument("--l_r", default=1e-3, type=float,
                    help="Specifies the learning rate of the Optimizer.")


def demo(variables=None):
    """
    This demo shows an example of Bayesian GP DMD on the minimum_jerk_trajectories data.
    :return: None
    """
    args = parser.parse_args()

    jax.config.update('jax_platform_name', args.device)
    jax.config.update('jax_enable_x64', args.x64)

    t_steps = jnp.arange(0, 1, 0.01)

    prng_handler = PRNGHandler(seed=args.seed)

    # ----------------------------------------------------------------------------------------------------------------
    # -------------------------------------------- Generate Data -----------------------------------------------------
    # ----------------------------------------------------------------------------------------------------------------

    y_init = jnp.deg2rad(jnp.array([[-10.], [50.]]))
    y_final = jnp.deg2rad(jnp.array([[50.], [-20.]]))
    data = MinimumJerk(y_init, y_final, t_steps, s_size=1, sigma=args.sigma, prng_handler=prng_handler)
    y = jnp.concatenate(data.transform, axis=1).T

    standarizer = Whitening(y.reshape(1, -1, order='F'))
    y = jnp.float64(standarizer().T.reshape(*y.shape, order='F'))

    y = y.reshape(data.transform.transpose((0, 2, 1)).shape)
    y = y[0]

    # ----------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------ Init ----------------------------------------------------------
    # ----------------------------------------------------------------------------------------------------------------

    iterations = args.iterations
    m = args.latent_dim
    number = 10

    theta = 1.
    gamma = 1/32
    i_points = 25

    t, n = y.shape

    _u, _s, _v = jnp.linalg.svd(y)
    x = y @ _v[:, :m]

    _idx = jnp.linspace(0, t, i_points, dtype=int)
    z = x[_idx] + jax.random.uniform(prng_handler.get_keys(1)[0], (i_points, m), minval=-1, maxval=1)

    alpha_0_init, beta_0_init = 1e5, 1.
    alpha_x_init, beta_x_init = 1e5, 1.
    alpha_y_init, beta_y_init = 1., 1e-5
    alpha_a_init, beta_a_init = 1e-3, 1.

    alpha_0, beta_0 = alpha_0_init, beta_0_init
    alpha_x, beta_x = alpha_x_init, beta_x_init
    alpha_y, beta_y = alpha_y_init, beta_y_init
    alpha_a, beta_a = alpha_a_init, beta_a_init

    mu_a, sigma_a = jnp.zeros((m, m)), jnp.eye(m)
    mu_u, sigma_u = y[i_points], jnp.eye(i_points)

    particles = jnp.zeros((t, number, m))

    @jit
    def kernel_fun(_param, _x, _y):
        i = _x.shape[1]
        j = _y.shape[1]
        inner = - 2 * jnp.einsum('ni, nj -> ij', _x, _y) + \
            jnp.ones((i, j)) * jnp.diag(jnp.einsum('ni, nj -> ij', _y, _y)) + \
            (jnp.ones((j, i)) * jnp.diag(jnp.einsum('ni, nj -> ij', _x, _x))).T
        return _param[1] * jnp.exp(-_param[0]/2 * inner)

    @jit
    def log_det(x1):
        return jnp.log(jnp.linalg.det(x1) + 1e-8)

    @jit
    def inv(matrix):
        return jnp.linalg.lstsq(matrix, jnp.eye(matrix.shape[0]))[0]

    @jax.partial(jit, static_argnums=(1,))
    def loss(_params, _kernel_fun, *args):
        _gamma, _theta, _z = _params
        _y, _x, _alpha_y, _beta_y = args
        _t, _n = _y.shape

        _kernel_zz = _kernel_fun((_gamma, _theta), _z.T, _z.T)
        _kernel_zz_inv = jnp.linalg.pinv(_kernel_zz)
        _kernel_tz = _kernel_fun((_gamma, _theta), _x.T, _z.T)
        _kernel_tt = _kernel_fun((_gamma, _theta), _x.T, _x.T)

        _sigma_y = _beta_y / _alpha_y
        _d_op = _kernel_tt - _kernel_tz @ _kernel_zz_inv @ _kernel_tz.T
        _lambda =  _sigma_y * jnp.eye(t) - _sigma_y ** 2 * _kernel_tz @ \
                   jnp.linalg.pinv(_kernel_zz + _sigma_y * _kernel_tz.T @ _kernel_tz)\
                   @ _kernel_tz.T

        _loss = - _sigma_y * _n / 2 * jnp.trace(_d_op) + _n/2 * log_det(_kernel_zz) \
                - _n / 2 * log_det(_kernel_zz + _sigma_y * _kernel_tz.T @ _kernel_tz) \
                - 1 / 2 * jnp.trace(_lambda @ _y @ _y.T)
        return - _loss

    from utilities.optimizer_lib import Adam
    optimizer = Adam(loss, l_r=args.l_r)

    @jit
    def _close_form_q_a(params, l_r, *args):
        x, alpha_x, beta_x, alpha_a, beta_a = args
        m, n = params[0].shape

        psi_1 = x[:-1].T @ ((alpha_x/beta_x) * jnp.eye(x[:-1].shape[0])) @ x[1:]
        psi_2 = x[:-1].T @ ((alpha_x/beta_x) * jnp.eye(x[:-1].shape[0])) @ x[:-1]

        i = jnp.eye(m)

        sigma_a = inv(alpha_a/beta_a * i + psi_2)
        mu_a = sigma_a @ psi_1
        return params[0] + l_r*(mu_a - params[0]), params[1] + l_r*(sigma_a - params[1])

    @jit
    def _close_form_q_0(params, l_r, *args):
        x, alpha_0_init, beta_0_init = args
        m = x.shape[1]

        alpha_0 = (m + 2* alpha_0_init)/2
        beta_0 =  2*beta_0_init + jnp.trace(x[0:1, :].T@x[0:1, :])

        return params[0] + l_r*(alpha_0 - params[0]), params[1] + l_r*(beta_0 - params[1])

    @jit
    def _close_form_q_x(params, l_r, *args):
        x, mu_a, sigma_a, alpha_x_init, beta_x_init = args
        t = x.shape[0]
        m = mu_a.shape[0]

        alpha_x = ((t-1)*m + 2 * alpha_x_init) / 2
        beta_x = 2 * beta_x_init \
                 + jnp.trace(x[1:, :].T @ x[1:, :] - 2 * x[:-1, :].T @ x[1:, :] @ mu_a.T
                             + x[:-1, :].T @ x[:-1, :] @ mu_a @ mu_a.T) \
                 + jnp.trace( x[:-1, :] @ (m * sigma_a) @ x[:-1, :].T)

        return params[0] + l_r * (alpha_x - params[0]), params[1] + l_r * (beta_x - params[1])

    @jit
    def _close_form_q_y(params, l_r, *args):
        y, x, z, gamma, theta, mu_u, sigma_u, alpha_y_init, beta_y_init = args
        t, n = x.shape

        _kernel_zz = kernel_fun((gamma, theta), z.T, z.T)
        _kernel_zz_inv = inv(_kernel_zz)
        _kernel_tz = kernel_fun((gamma, theta), x.T, z.T)
        _kernel_tt = kernel_fun((gamma, theta), x.T, x.T)

        c_op = _kernel_tz @ _kernel_zz_inv
        d_op = _kernel_tt - _kernel_tz @ _kernel_zz_inv @ _kernel_tz.T

        alpha_y = (t*n + 2* alpha_y_init)/2
        beta_y =  2*beta_y_init + jnp.trace((y - c_op @ mu_u)@(y - c_op @ mu_u).T) \
                  + n*jnp.trace(d_op + c_op @ sigma_u @ c_op.T)

        return params[0] + l_r*(alpha_y - params[0]), params[1] + l_r*(beta_y - params[1])

    @jit
    def _close_form_q_u(params, l_r, *args):
        y, x, z, gamma, theta, alpha_y, beta_y = args

        _kernel_zz = kernel_fun((gamma, theta), z.T, z.T)
        _kernel_zz_inv = inv(_kernel_zz)
        _kernel_tz = kernel_fun((gamma, theta), x.T, z.T)

        psi_3 = _kernel_tz.T @ ((alpha_y/beta_y) * jnp.eye(_kernel_tz.shape[0])) @  y
        psi_4 = _kernel_tz.T @ ((alpha_y/beta_y) * jnp.eye(_kernel_tz.shape[0])) @  _kernel_tz

        sigma_u = inv(_kernel_zz_inv + _kernel_zz_inv @ psi_4 @ _kernel_zz_inv)
        mu_u = sigma_u @ _kernel_zz_inv @ psi_3
        return params[0] + l_r*(mu_u - params[0]), params[1] + l_r*(sigma_u - params[1])

    @jit
    def _close_form_lambda_a(params, l_r, *args):
        mu_a, sigma_a, alpha_a_init, beta_a_init = args
        m, _ = mu_a.shape

        alpha_a = 2*alpha_a_init + m**2
        beta_a = 2*beta_a_init + jnp.trace(mu_a @ mu_a.T + m * sigma_a)
        return params[0] + l_r*(alpha_a - params[0]), params[1] + l_r*(beta_a - params[1])

    # ---------------------------------------- Sequentiell Monte Carlo -------------------------------------------------
    @jax.partial(jit, static_argnums=(0,))
    def p_g(_kernel_fun, _particles, label, *args):
        gamma, theta, z, mu_u, sigma_u, _kernel_zz_inv, sigma_y = args
        _kernel_tz = _kernel_fun((gamma, theta), _particles.T, z.T)
        _kernel_tt = _kernel_fun((gamma, theta), _particles.T, _particles.T)
        c_op = _kernel_tz @ _kernel_zz_inv
        nrml = MultivariateNormal(loc=c_op @ mu_u, covariance_matrix= sigma_y * jnp.eye(n))
        return jnp.exp(nrml.log_prob(label))+ 1e-8

    @jit
    def pi_x(_key, _particles, mu_a, sigma_x, sigma_a):
        m = sigma_a.shape[0]
        sigma = sigma_x * jnp.eye(m)
        nrml = MultivariateNormal(loc=_particles @ mu_a, covariance_matrix=sigma)
        return nrml.sample(_key)

    # ----------------------------------------------------------------------------------------------------------------
    # ---------------------------------------------- Training --------------------------------------------------------
    # ----------------------------------------------------------------------------------------------------------------

    ll_values = []
    opt_state = optimizer.opt_init([gamma, theta, z])

    mu_u, sigma_u = _close_form_q_u([mu_u, sigma_u], 1, y, x, z, gamma, theta, alpha_y, beta_y)
    mu_a, sigma_a = _close_form_q_a([mu_a, sigma_a], 1, x, alpha_x, beta_x, alpha_a, beta_a)
    alpha_0, beta_0 = _close_form_q_0([alpha_0, beta_0], 1, x, alpha_0_init, beta_0_init)
    alpha_x, beta_x = _close_form_q_x([alpha_x, beta_x], 1, x, mu_a, sigma_a, alpha_x_init, beta_x_init)
    alpha_y, beta_y = _close_form_q_y([alpha_y, beta_y], 1, y, x, z, gamma, theta, mu_u, sigma_u,
                                      alpha_y_init, beta_y_init)
    alpha_a, beta_a = _close_form_lambda_a([alpha_a, beta_a], 1, mu_a, sigma_a, alpha_a_init, beta_a_init)

    for _ in tqdm(range(iterations)):
        # ------------------------------------------------ E - step ---------------------------------------------------
        # ---------------------------------------- Prediction Latent Space --------------------------------------------

        kernel_zz_inv = inv(kernel_fun((gamma, theta), z.T, z.T))
        args = (gamma, theta, z, mu_u, sigma_u, kernel_zz_inv, beta_y / alpha_y)
        particles *= 0
        nrml = MultivariateNormal(loc=x[0], covariance_matrix=(beta_0 / alpha_0) * jnp.eye(m))
        particles = index_update(particles, index[0],
                                 nrml.sample(prng_handler.get_keys(1)[0], sample_shape=(number,)))
        weights = jnp.ones(number) / number
        weights = weights * p_g(kernel_fun, particles[0], y[0], *args)
        weights /= jnp.sum(weights)

        for i in range(1, t):
            particles = index_update(particles, index[i], pi_x(prng_handler.get_keys(1)[0], particles[i - 1],
                                                               mu_a, beta_x / alpha_x, sigma_a))
            weights = weights * p_g(kernel_fun, particles[i], y[i], *args)
            weights /= jnp.sum(weights)
            weights = jnp.clip(weights, a_min=1e-8)
        x = jnp.sum(weights[None, :, None] * particles, axis=1)
        x = jnp.clip(x, a_min=-1e+10, a_max=1e+10)

        # ------------------------------------------ Closed From Solutions --------------------------------------------

        mu_u, sigma_u = _close_form_q_u([mu_u, sigma_u], 1, y, x, z, gamma, theta, alpha_y, beta_y)
        mu_a, sigma_a = _close_form_q_a([mu_a, sigma_a], 1, x, alpha_x, beta_x, alpha_a, beta_a)
        alpha_0, beta_0 = _close_form_q_0([alpha_0, beta_0], 1, x, alpha_0_init, beta_0_init)
        alpha_x, beta_x = _close_form_q_x([alpha_x, beta_x], 1, x, mu_a, sigma_a, alpha_x_init, beta_x_init)
        alpha_y, beta_y = _close_form_q_y([alpha_y, beta_y], 1, y, x, z, gamma, theta, mu_u, sigma_u,
                                          alpha_y_init, beta_y_init)
        alpha_a, beta_a = _close_form_lambda_a([alpha_a, beta_a], 1, mu_a, sigma_a, alpha_a_init, beta_a_init)

        # -------------------------------------------------------------------------------------------------------------
        # ---------------------------------------------- M - Step -----------------------------------------------------
        args = [y, x, alpha_y, beta_y]
        params = [gamma, theta, z]
        params, opt_state, value = optimizer.step(params, opt_state, kernel_fun, *args)
        gamma, theta, z = params
        ll_values.append(value)

    # ----------------------------------------------------------------------------------------------------------------
    # ----------------------------------------- Prediction Latent Space ----------------------------------------------
    # ----------------------------------------------------------------------------------------------------------------

    kernel_zz_inv = inv(kernel_fun((gamma, theta), z.T, z.T))
    args = (gamma, theta, z, mu_u, sigma_u, kernel_zz_inv, beta_y / alpha_y)
    particles *= 0
    nrml = MultivariateNormal(loc=x[0], covariance_matrix=(beta_0 / alpha_0) * jnp.eye(m))
    particles = index_update(particles, index[0],
                             nrml.sample(prng_handler.get_keys(1)[0], sample_shape=(number,)))
    weights = jnp.ones(number) / number
    weights = weights * p_g(kernel_fun, particles[0], y[0], *args)
    weights /= jnp.sum(weights)

    for i in range(1, t):
        particles = index_update(particles, index[i], pi_x(prng_handler.get_keys(1)[0], particles[i - 1],
                                                           mu_a, beta_x / alpha_x, sigma_a))
        weights = weights * p_g(kernel_fun, particles[i], y[i], *args)
        weights /= jnp.sum(weights)
        weights = jnp.clip(weights, a_min=1e-8)
    x = jnp.sum(weights[None, :, None] * particles, axis=1)
    x = jnp.clip(x, a_min=-1e+10, a_max=1e+10)

    # ----------------------------------------------------------------------------------------------------------------
    # --------------------------------------- Prediction Observation Space -------------------------------------------
    # ----------------------------------------------------------------------------------------------------------------

    kernel_zz_inv = inv(kernel_fun((gamma, theta), z.T, z.T))
    kernel_tz = kernel_fun((gamma, theta), x.T, z.T)
    kernel_tt = kernel_fun((gamma, theta), x.T, x.T)
    c_op = kernel_tz @ kernel_zz_inv
    mu = c_op @ mu_u
    cov = kernel_tt - kernel_tz @ kernel_zz_inv @ kernel_tz.T

    nrml = MultivariateNormal(loc=mu.T, covariance_matrix=pos_def(cov))
    num_samples2 = 20
    _samples = nrml.sample(prng_handler.get_keys(1)[0], sample_shape=(num_samples2,))

    # ----------------------------------------------------------------------------------------------------------------
    # ----------------------------------------------- Visualization --------------------------------------------------
    # ----------------------------------------------------------------------------------------------------------------

    fig = plt.figure()
    grid = fig.add_gridspec(6, 2)

    ax11 = fig.add_subplot(grid[0, :])
    ax11.set_title(f"Target Data")
    ax11.set_xlabel("Time")
    ax11.set_ylabel("$\Theta$")
    plt.gca().set_prop_cycle(None)
    ax11.plot(t_steps, y)

    ax12 = fig.add_subplot(grid[1, :])
    ax12.set_title(f"Target Data")
    ax12.set_xlabel("Time")
    ax12.set_ylabel("$\Theta$")
    plt.gca().set_prop_cycle(None)
    ax12.plot(t_steps, mu)

    ax13 = fig.add_subplot(grid[2, :])
    ax13.set_title(f"Samples")
    ax13.set_xlabel("Time")
    ax13.set_ylabel("$\Theta$")
    for s in _samples:
        plt.gca().set_prop_cycle(None)
        ax13.plot(t_steps, jnp.real(s.T))

    ax140 = fig.add_subplot(grid[3, 0])
    ax140.set_title(f"Latent_dim")
    ax140.set_xlabel("Time")
    ax140.set_ylabel("$\Theta$")
    ax140.plot(t_steps, x)

    ax141 = fig.add_subplot(grid[3, 1])
    ax141.set_title(f"Latent_dim_samples")
    ax141.set_xlabel("Time")
    ax141.set_ylabel("$\Theta$")
    for i in range(number):
        ax141.plot(t_steps, particles[:, i])

    ax150 = fig.add_subplot(grid[4, 0])
    ax150.set_title(f"Inducing Variables")
    ax150.set_xlabel("Time")
    ax150.set_ylabel("$\mu_{u}$")
    ax150.plot(t_steps[_idx], mu_u)

    ax151 = fig.add_subplot(grid[4, 1])
    ax151.set_title(f"Inducing Inputs")
    ax151.set_xlabel("Time")
    ax151.set_ylabel("$z$")
    ax151.plot(t_steps[_idx], z)

    ax16 = fig.add_subplot(grid[5, :])
    ax16.set_title("Log_Likelihood")
    ax16.set_xlabel("Sample")
    ax16.set_ylabel("$log_likelihood value$")
    ax16.plot(jnp.arange(len(ll_values)), ll_values)

    # Figure
    num_samples2 = 100
    _samples = nrml.sample(prng_handler.get_keys(1)[0], sample_shape=(num_samples2,))
    fig2 = plt.figure()
    grid2 = fig2.add_gridspec(3, 2)
    ax200 = fig2.add_subplot(grid2[0, 0])
    ax200.set_title(f"Position 1. Joint")
    ax200.set_xlabel("Time")
    ax200.set_ylabel("$x$")
    for s in _samples:
        ax200.plot(t_steps, jnp.real(s[0]), 'b', alpha=0.05)
    ax200.plot(t_steps, mu[:, 0], 'b', label='Mean')
    ax200.plot(t_steps, y[:, 0], '--r', label='Data')
    ax200.legend()

    ax201 = fig2.add_subplot(grid2[0, 1])
    ax201.set_title(f"Position 2. Joint")
    ax201.set_xlabel("Time")
    ax201.set_ylabel("$x$")
    for s in _samples:
        ax201.plot(t_steps, jnp.real(s[1]), 'b', alpha=0.05)
    ax201.plot(t_steps, mu[:, 1], 'b', label='Mean')
    ax201.plot(t_steps, y[:, 1], '--r', label='Data')
    ax201.legend()

    ax210 = fig2.add_subplot(grid2[1, 0])
    ax210.set_title(f"Velocity 1. Joint")
    ax210.set_xlabel("Time")
    ax210.set_ylabel("$v$")
    plt.gca().set_prop_cycle(None)
    for s in _samples:
        ax210.plot(t_steps, jnp.real(s[2]), 'b', alpha=0.05)
    ax210.plot(t_steps, mu[:, 2], 'b', label='Mean')
    ax210.plot(t_steps, y[:, 2], '--r', label='Data')
    ax210.legend()

    ax211 = fig2.add_subplot(grid2[1, 1])
    ax211.set_title(f"Velocity 2. Joint")
    ax211.set_xlabel("Time")
    ax211.set_ylabel("$v$")
    for s in _samples:
        ax211.plot(t_steps, jnp.real(s[3]), 'b', alpha=0.05)
    ax211.plot(t_steps, mu[:, 3], 'b', label='Mean')
    ax211.plot(t_steps, y[:, 3], '--r', label='Data')
    ax211.legend()

    ax220 = fig2.add_subplot(grid2[2, 0])
    ax220.set_title(f"Acceleration 1. Joint")
    ax220.set_xlabel("Time")
    ax220.set_ylabel("$a$")
    for s in _samples:
        ax220.plot(t_steps, jnp.real(s[4]), 'b', alpha=0.05)
    ax220.plot(t_steps, mu[:, 4], 'b', label='Mean')
    ax220.plot(t_steps, y[:, 4], '--r', label='Data')
    ax220.legend()

    ax221 = fig2.add_subplot(grid2[2, 1])
    ax221.set_title(f"Acceleration 2. Joint")
    ax221.set_xlabel("Time")
    ax221.set_ylabel("$a$")
    for s in _samples:
        ax221.plot(t_steps, jnp.real(s[5]), 'b', alpha=0.05)
    ax221.plot(t_steps, mu[:, 5], 'b', label='Mean')
    ax221.plot(t_steps, y[:, 5], '--r', label='Data')
    ax221.legend()
    plt.show()

    mse_mean = jnp.trace((mu-y)@(mu-y).T)/mu.shape[0]
    mse_samples = []
    for s in _samples:
        mse_samples.append(jnp.trace((s.T - y)@(s.T - y).T) / mu.shape[0])
    delta_mse = jnp.array(mse_samples) - mse_mean
    mse_std = jnp.sqrt(jnp.sum(delta_mse**2)/len(mse_samples))
    print(f'The Mean Squared Error: {jnp.absolute(mse_mean)}')
    print(f'The Standard Deviation of the Mean Squared Error: {jnp.absolute(mse_std)}')


if __name__ == '__main__':
    demo()
