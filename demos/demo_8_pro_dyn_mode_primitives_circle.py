from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import argparse

from datasets import CircleShapes
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
parser.add_argument("--samples", default=5, type=int,
                    help="Set the number of sampled trajectories.")

parser.add_argument("--latent_dim", default=3, type=int,
                    help="Dimensionality of the latent space.")
parser.add_argument("--iterations", default=50000, type=int,
                    help="Specifies the iterations of the Optimizer.")
parser.add_argument("--l_r", default=1e-3, type=float,
                    help="Specifies the learning rate of the Optimizer.")


def demo(variables=None):
    """
    This demo shows an example of the probabilistic dynamic mode primitives on a circle shape dataset
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

    data = CircleShapes(t_steps, s_size=args.samples, prng_handler=prng_handler)
    y = jnp.concatenate(data.transform,axis=1).T

    standarizer = Whitening(y.reshape(1, -1, order='F'))
    y = jnp.float64(standarizer().T.reshape(*y.shape, order='F'))

    y = y.reshape(data.transform.transpose((0, 2, 1)).shape)

    # ----------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------ Init ----------------------------------------------------------
    # ----------------------------------------------------------------------------------------------------------------

    iterations = args.iterations
    m = args.latent_dim
    number = 10
    s = args.samples

    theta = 1.
    gamma = 1/16

    alpha_x = 1e3
    beta_x = 1.
    alpha_0 = 1e3
    beta_0 = 1.
    alpha_y = 1.
    beta_y = 1e-3

    alpha_a = 1e-6
    beta_a = jnp.eye(m) * 1e-6

    lambda_x = alpha_x/beta_x
    lambda_0 = alpha_0/beta_0
    lambda_y = alpha_y/beta_y

    mu_a = jnp.zeros((m, m))
    lambda_a = jnp.zeros((m, m, m))

    mu_a_s = jnp.zeros((s, m, m))
    sigma_a_s = jnp.zeros((s, m, m, m))

    s, t, n = y.shape
    x = jnp.ones(shape=(s, t, m))

    @jit
    def kernel_fun(_param, _x, _y):
        i = _x.shape[1]
        j = _y.shape[1]
        inner = - 2 * jnp.einsum('ni, nj -> ij', _x, _y) + \
            jnp.ones((i, j)) * jnp.diag(jnp.einsum('ni, nj -> ij', _y, _y)) + \
            (jnp.ones((j, i)) * jnp.diag(jnp.einsum('ni, nj -> ij', _x, _x))).T
        return _param[1] * jnp.exp(-_param[0]/2 * inner)

    @jit
    def inv(matrix):
        return jnp.linalg.lstsq(matrix, jnp.eye(matrix.shape[0]))[0]

    @jax.partial(jit, static_argnums=(1,))
    def loss(params, kernel_fun, *args):
        _gamma, _theta, x, lambda_y = params
        y, mu_a_s, sigma_a_s, lambda_0, lambda_x, alpha_y, beta_y = args

        s, t, n = y.shape

        loss = 0
        for s_iter in range(s):
            y_outer = jnp.einsum('kd, hd -> kh', y[s_iter], y[s_iter])

            kernel = kernel_fun((_gamma, _theta), x[s_iter].T, x[s_iter].T)
            kernel += jnp.eye(*kernel.shape) / lambda_y
            chol = jnp.linalg.cholesky(kernel)
            cholinv = jnp.linalg.inv(chol)
            kernelinv = cholinv.T @ cholinv

            loss += 1 / 2 * (n * jnp.sum(jnp.log(jnp.linalg.det(kernel))) + jnp.einsum('nn -> ', kernelinv @ y_outer)
                             + lambda_x * jnp.trace(x[s_iter, 1:, :].T @ x[s_iter, 1:, :]
                                                    - 2 * x[s_iter, :-1, :].T @ x[s_iter, 1:, :] @ mu_a_s[s_iter].T
                                                    + x[s_iter, :-1, :].T @ x[s_iter, :-1, :] @ mu_a_s[s_iter] @ mu_a_s[s_iter].T
                                                    + x[s_iter, :-1, :].T @ x[s_iter, :-1, :] @ jnp.sum(sigma_a_s[s_iter], axis=0))
                             + lambda_0*jnp.trace(x[s_iter, 0:1, :].T @x [s_iter, 0:1, :])
                             - (alpha_y - 1) * jnp.log(lambda_y) + beta_y*lambda_y)
        return loss

    from utilities.optimizer_lib import Adam
    optimizer = Adam(loss, l_r=args.l_r)

    @jit
    def _close_form_q_a(params, l_r, *args):
        x, mu_a, lambda_a, lamda_x = args

        s, t, m = x.shape

        mu_a_s = jnp.zeros_like(params[0])
        sigma_a_s = jnp.zeros_like(params[1])

        for s_iter in range(s):
            psi_1 = x[s_iter, :-1].T @ ((alpha_x/beta_x) * jnp.eye(t-1)) @ x[s_iter, 1:]
            psi_2 = x[s_iter, :-1].T @ ((alpha_x/beta_x) * jnp.eye(t-1)) @ x[s_iter, :-1]

            for m_iter in range(m):
                sigma_a_s =  index_update(sigma_a_s, index[s_iter, m_iter], inv(lambda_a[m_iter] + psi_2))

                mu_a_s = index_update(mu_a_s, index[s_iter, :, m_iter],
                                        sigma_a_s[s_iter, m_iter] @ (psi_1[:, m_iter] + lambda_a[m_iter]@mu_a[:, m_iter]))
        return params[0] + l_r*(mu_a_s - params[0]), params[1] + l_r*(sigma_a_s - params[1])

    @jit
    def _close_form_mu_a(params, l_r, mu_a_s):
        s, _, _ = mu_a_s.shape

        return params + l_r*(jnp.sum(mu_a_s, axis=0)/s - params)

    @jit
    def _close_form_lambda_a(params, l_r, *args):
        mu_a, mu_a_s, sigma_a_s, alpha_a, beta_a = args
        m, _ = mu_a.shape

        lambda_a = jnp.zeros_like(params)
        delta = mu_a_s - mu_a

        for m_iter in range(m):
            result = delta[:, m_iter, :].T @ delta[:, m_iter, :] +  jnp.sum(sigma_a_s[:, m_iter], axis=0) + beta_a
            lambda_a = index_update(lambda_a, index[m_iter], inv(result) * (s + alpha_a))
        return params + l_r*(lambda_a - params)

    @jit
    def _close_form_lambda_0(params, l_r, *args):
        x, alpha_0, beta_0 = args
        s, _, m = x.shape

        lambda_0 = (s*m + 2 * alpha_0 - 2) / (jnp.trace(x[:, 0, :].T@x[:, 0, :]) + 2 * beta_0)
        return params + l_r*(lambda_0 - params)

    @jit
    def _close_form_lambda_x(params, l_r, *args):
        x, mu_a_s, sigma_a_s, alpha_x, beta_x = args
        s, t, m = x.shape

        result = 0
        for s_iter in range(s):
            result += jnp.trace(x[s_iter, 1:, :].T @ x[s_iter, 1:, :]
                                - 2 * x[s_iter, :-1, :].T @ x[s_iter, 1:, :] @ mu_a_s[s_iter].T
                                + x[s_iter, :-1, :].T @ x[s_iter, :-1, :] @ mu_a_s[s_iter] @ mu_a_s[s_iter].T
                                + x[s_iter, :-1, :].T @ x[s_iter, :-1, :] @ jnp.sum(sigma_a_s[s_iter], axis=0))

        lambda_x = (s*(t-1)*m + 2 * alpha_x - 2) / (result + 2 * beta_x)
        return params + l_r*(lambda_x - params)

    # ----------------------------------------------------------------------------------------------------------------
    # ---------------------------------------------- Training --------------------------------------------------------
    # ----------------------------------------------------------------------------------------------------------------

    ll_values = []
    opt_state = optimizer.opt_init([gamma, theta, x, lambda_y])

    for _iter in tqdm(range(iterations)):
        mu_a_s, sigma_a_s = _close_form_q_a([mu_a_s, sigma_a_s], 1, x, mu_a, lambda_a, lambda_x)
        lambda_0 = _close_form_lambda_0(lambda_0, 1, x, alpha_0, beta_0)
        lambda_x = _close_form_lambda_x(lambda_x, 1, x, mu_a_s, sigma_a_s, alpha_x, beta_x)

        mu_a = _close_form_mu_a(mu_a, 1, mu_a_s)
        lambda_a = _close_form_lambda_a(lambda_a, 1, mu_a, mu_a_s, sigma_a_s, alpha_a, beta_a)

        args = [y, mu_a_s, sigma_a_s, lambda_0, lambda_x, alpha_y, beta_y]
        params, opt_state, value = optimizer.step([gamma, theta, x, lambda_y], opt_state, kernel_fun, *args)
        gamma, theta, x, lambda_y = params
        ll_values.append(value)

    # ----------------------------------------------------------------------------------------------------------------
    # ----------------------------------------- Prediction Latent Space ----------------------------------------------
    # ----------------------------------------------------------------------------------------------------------------

    def _vander(x: jnp.ndarray, n: int, pow: float=1) -> jnp.ndarray:
        return jnp.column_stack([x ** (i * pow) for i in range(n)])

    a_tilde = mu_a

    a_tilde = a_tilde.T
    mu, phi = jnp.linalg.eig(a_tilde)
    x_0 = jnp.sum(x[:, 0, :], axis=0)/s
    b = jnp.linalg.lstsq(phi, x_0, rcond=None)[0]

    time_behaviour = _vander(mu, len(t_steps))
    x_latent = (phi @ jnp.diag(b) @ time_behaviour).T
    x_latent2 = x[0]

    x_latent = jnp.clip(x_latent, a_min=-5e+0, a_max=5e+0)

    # ----------------------------------------------------------------------------------------------------------------
    # --------------------------------------- Prediction Observation Space -------------------------------------------
    # ----------------------------------------------------------------------------------------------------------------

    labels_idx = [0, 33, 66, 99]
    labels_y = y[0, labels_idx, :]

    K11 = jnp.nan_to_num(kernel_fun((gamma, theta), x_latent[labels_idx, :].T, x_latent[labels_idx, :].T))
    K12 = jnp.nan_to_num(kernel_fun((gamma, theta), x_latent[labels_idx, :].T, x_latent.T))
    K21 = K12.T
    K22 = jnp.nan_to_num(kernel_fun((gamma, theta), x_latent.T, x_latent.T))
    K_11_inv = inv(K11 + jnp.eye(*K11.shape) / lambda_y)

    mean = jnp.nan_to_num(K21 @ K_11_inv @ labels_y)
    sigma = jnp.nan_to_num(K22 - K21 @ K_11_inv @ K12)

    mvnrml = MultivariateNormal(loc=mean.T, covariance_matrix=pos_def(sigma))
    _samples = jnp.nan_to_num(mvnrml.sample(prng_handler.get_keys(1)[0], sample_shape=(number,)))

    K11 = jnp.nan_to_num(kernel_fun((gamma, theta), x_latent2[labels_idx, :].T, x_latent2[labels_idx, :].T))
    K12 = jnp.nan_to_num(kernel_fun((gamma, theta), x_latent2[labels_idx, :].T, x_latent2.T))
    K21 = K12.T
    K22 = jnp.nan_to_num(kernel_fun((gamma, theta), x_latent2.T, x_latent2.T))
    K_11_inv = inv(K11 + jnp.eye(*K11.shape) / lambda_y)

    mean2 = jnp.nan_to_num(K21 @ K_11_inv @ labels_y)
    sigma2 = jnp.nan_to_num(K22 - K21 @ K_11_inv @ K12)

    mvnrml2 = MultivariateNormal(loc=mean2.T, covariance_matrix=pos_def(sigma2))
    _samples2 = jnp.nan_to_num(mvnrml2.sample(prng_handler.get_keys(1)[0], sample_shape=(number,)))

    # ----------------------------------------------------------------------------------------------------------------
    # ----------------------------------------------- Visualization --------------------------------------------------
    # ----------------------------------------------------------------------------------------------------------------

    fig = plt.figure()
    grid = fig.add_gridspec(5, 2)

    ax11 = fig.add_subplot(grid[0, :])
    ax11.set_title(f"Target Data")
    ax11.set_xlabel("Time")
    ax11.set_ylabel("$\Theta$")
    plt.gca().set_prop_cycle(None)
    for s_iter in range(s):
        ax11.plot(t_steps, y[s_iter])

    ax12 = fig.add_subplot(grid[1, 0])
    ax12.set_title(f"Target Data of recreated")
    ax12.set_xlabel("Time")
    ax12.set_ylabel("$\Theta$")
    plt.gca().set_prop_cycle(None)
    ax12.plot(t_steps, mean)

    ax121 = fig.add_subplot(grid[1, 1])
    ax121.set_title(f"Target Data of original x")
    ax121.set_xlabel("Time")
    ax121.set_ylabel("$\Theta$")
    plt.gca().set_prop_cycle(None)
    ax121.plot(t_steps, mean2)

    ax13 = fig.add_subplot(grid[2, 0])
    ax13.set_title(f"Samples")
    ax13.set_xlabel("Time")
    ax13.set_ylabel("$\Theta$")
    for s_iter in _samples:
        plt.gca().set_prop_cycle(None)
        ax13.plot(t_steps, jnp.real(s_iter.T))

    ax131 = fig.add_subplot(grid[2, 1])
    ax131.set_title(f"Samples of recreated")
    ax131.set_xlabel("Time")
    ax131.set_ylabel("$\Theta$")
    for s_iter2 in _samples2:
        plt.gca().set_prop_cycle(None)
        ax131.plot(t_steps, jnp.real(s_iter2.T))

    ax14 = fig.add_subplot(grid[3, 0])
    ax14.set_title(f"Latent_dim")
    ax14.set_xlabel("Time")
    ax14.set_ylabel("$\Theta$")
    ax14.plot(t_steps, jnp.real(x_latent), '--')

    ax141 = fig.add_subplot(grid[3, 1])
    ax141.set_title(f"Latent_dim")
    ax141.set_xlabel("Time")
    ax141.set_ylabel("$\Theta$")
    for s_iter in range(s):
        plt.gca().set_prop_cycle(None)
        ax141.plot(t_steps, jnp.real(x[s_iter]), '--')

    ax15 = fig.add_subplot(grid[4, :])
    ax15.set_title("Log_Likelihood")
    ax15.set_xlabel("Sample")
    ax15.set_ylabel("$log_likelihood value$")
    ax15.plot(jnp.arange(len(ll_values)), ll_values)

    fig2 = plt.figure()
    grid2 = fig2.add_gridspec(1,1)
    ax2 = fig2.add_subplot(grid2[0,0], projection='3d')
    _samples = jnp.nan_to_num(mvnrml.sample(prng_handler.get_keys(1)[0], sample_shape=(100,)))
    for s_iter in _samples:
        ax2.plot(s_iter[0], s_iter[1], s_iter[2], color='blue', alpha=0.05)
    ax2.plot(mean[:, 0], mean[:, 1], mean[:, 2], label='Mean', color='blue')
    for i in range(y.shape[0]):
        ax2.plot(y[i, :, 0], y[i, :, 1], y[i, :, 2], label='Data', color='red')
    ax2.set_title("GPDMD Eight-Shape Dataset")
    ax2.set_xlabel("$x$")
    ax2.set_ylabel("$y$")
    ax2.set_zlabel("$z$")
    ax2.legend()
    plt.show()

    y_mean = jnp.sum(y, axis=0)/y.shape[0]
    mse_mean = jnp.trace((mean-y_mean)@(mean-y_mean).T)/mean.shape[0]
    mse_samples = []
    for s in _samples:
        mse_samples.append(jnp.trace((s.T - y_mean)@(s.T - y_mean).T) / mean.shape[0])
    delta_mse = jnp.array(mse_samples) - mse_mean
    mse_std = jnp.sqrt(jnp.sum(delta_mse**2)/len(mse_samples))
    print(f'The Mean Squared Error: {jnp.absolute(mse_mean)}')
    print(f'The Standard Deviation of the Mean Squared Error: {jnp.absolute(mse_std)}')


if __name__ == '__main__':
    demo()