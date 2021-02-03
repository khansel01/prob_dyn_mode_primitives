import jax
import jax.numpy as jnp
import jax.random as random
import matplotlib.pyplot as plt
import argparse

from jax import jit
from jax.ops import index, index_update
from jax.experimental import optimizers as opt
from datasets import MinimumJerk
from models.dmd_models import *
from utilities.utils import data2snapshots
from utilities.prng_handler import PRNGHandler
from tqdm import tqdm


parser = argparse.ArgumentParser(description="Test")
parser.add_argument("--device", default="cpu", type=str,
                        help="Specify the device on which Jax works. Possible devices are cpu or gpu (if available).")
parser.add_argument("--x64", default=True, type=bool,
                        help="Defines the float type used by Jax.")

parser.add_argument("--dmd", default="Probabilistic", type=str,
                        help="Specifies the DMD method used. Possible Methods are "
                             "Standard, Exact, FB, tls, Kernel, Probabilistic, Bayesian")

parser.add_argument("--seed", default=11, type=int,
                        help="Defines the seed of the pseudo random number generator.")
parser.add_argument("--samples", default=1, type=int,
                        help="Set the number of sampled trajectories.")
parser.add_argument("--time_delay", default=1, type=int,
                        help="Set an integer to define the time shift used to create delayed coordinates.")
parser.add_argument("--axis", default=1, type=int,
                        help="Set the axis along which the trajectories will be concatenated.")

parser.add_argument("--sigma", default=0.1, type=float,
                        help="Set the variability in the sampled data.")
parser.add_argument("--gibbs_iter", default=5000, type=int,
                        help="Specifies the iterations of the Gibbs sampler.")
parser.add_argument("--gibbs_burn_in", default=1000, type=int,
                        help="Specifies the Burn-in phase of the Gibbs sampler.")

parser.add_argument("--latent_dim", default=2, type=int,
                        help="Dimensionality of the latent space.")
parser.add_argument("--alpha", default=1e-3, type=float,
                        help="Initial shape of the gamma distribution.")
parser.add_argument("--beta", default=1e-3, type=float,
                        help="Initial rate of the gamma distribution.")


def demo():
    """
    This demo shows an example of the exact dynamic mode decomposition on the minimum_jerk_trajectories data.
    :return: None
    """
    args = parser.parse_args()

    jax.config.update('jax_platform_name', args.device)
    jax.config.update('jax_enable_x64', args.x64)

    t_steps = jnp.arange(0, 1, 0.01)
    iterations = 50000
    labels = 4

    prng_handler = PRNGHandler(seed=args.seed)

    # Generate Data
    x_init = jnp.deg2rad(jnp.array([[-10.], [50.]]))
    x_final = jnp.deg2rad(jnp.array([[50.], [-20.]]))
    data = MinimumJerk(x_init, x_final, t_steps, s_size=args.samples, sigma=args.sigma, prng_handler=prng_handler)
    x0, x1 = data2snapshots(data.transform, t_delay=args.time_delay, axis=args.axis)

    x = jnp.hstack((x0, x1[:, -1].reshape(-1, 1))).T

    m, n = x.shape
    latent_dim = 5

    sigma_x = 1.
    sigma_y = 1e-2
    sigma_a = 1e-2

    a_tilde = random.normal(key = prng_handler.get_keys(1)[0], shape=(latent_dim, latent_dim))
    z = jnp.ones(shape=(m, latent_dim))


    @jit
    def loss(params, *args):
        _z, _sigma_y = params
        _a_tilde, _sigma_x, _sigma_a, _x_outer = args

        kernel = _z @ _z.T
        kernel += jnp.eye(*kernel.shape) * _sigma_y
        chol = jnp.linalg.cholesky(kernel)
        cholinv = jnp.linalg.inv(chol)
        kernelinv = cholinv.T @ cholinv

        loss = 1 / 2 * (m * latent_dim * jnp.log(_sigma_x) + latent_dim ** 2 * jnp.log(_sigma_a)
                        + (1 / _sigma_a) * jnp.trace(_a_tilde @ _a_tilde.T)
                        + n * jnp.sum(jnp.log(jnp.linalg.det(kernel))) + jnp.einsum('nn -> ', kernelinv @ _x_outer)
                        + (1 / _sigma_x) * jnp.trace(_z.T @ _z - 2 * _z[:-1, :].T @ _z[1:, :] @ _a_tilde.T
                                                     + _z[:-1, :].T @ _z[:-1, :] @ _a_tilde @ _a_tilde.T))
        return loss


    class Adam(object):
        def __init__(self, loss: jnp.function, params: list, **kwargs):
            self.loss = loss

            self.l_r = kwargs.get("l_r", 1e-3)
            self.b1 = kwargs.get("b1", 0.9)
            self.b2 = kwargs.get("b2", 0.999)
            self.eps = kwargs.get("eps", 1e-8)

            self.opt_init, self.opt_update, self.get_params = opt.adam(self.l_r, b1=self.b1, b2=self.b2, eps=self.eps)

            self.opt_state_init = self.opt_init(params)

        @jax.partial(jit, static_argnums=(0,))
        def step(self, _params, opt_state, *args):
            value, grads = jax.value_and_grad(self.loss)(_params, *args)
            opt_state = self.opt_update(0, grads, opt_state)
            return self.get_params(opt_state), opt_state, value


    @jit
    def close_form_updates(_params, *args):
        _, _sigma_a = _params
        _z, _sigma_x = args

        _a_tilde = jnp.linalg.pinv(_sigma_x / _sigma_a + _z[:-1, :].T @ _z[:-1, :]) @ _z[:-1, :].T @ _z[1:, :]

        _sigma_a = 1 / (latent_dim ** 2 - 1) * jnp.trace(_a_tilde @ _a_tilde.T)
        return _a_tilde, _sigma_a


    def train(optimizer, _iter, _params, *args):
        _a_tilde, _z, _sigma_y, _sigma_a = _params
        _x, _sigma_x = args
        _opt_state = optimizer.opt_state_init

        loss_list = []

        _x_outer = jnp.einsum('kd, hd -> kh', _x, _x)

        for i in tqdm(range(_iter)):
            args = [_a_tilde, _sigma_x, _sigma_a, _x_outer]
            _params, _opt_state, _value = adam.step([_z, _sigma_y], _opt_state, *args)
            _z, _sigma_y = _params
            _sigma_y = jnp.maximum(_sigma_y, 1e-2)

            args = [_z, _sigma_x]
            _a_tilde, _sigma_a = close_form_updates([_a_tilde, _sigma_a], *args)
            _sigma_a = jnp.maximum(_sigma_a, 1e-2)

            loss_list.append(_value)

        return [_a_tilde, _z, _sigma_y, _sigma_a], loss_list

    params = [z, sigma_y]

    l_r = 1e-3
    b1 = 0.9
    b2 = 0.999
    eps = 1e-8
    adam = Adam(loss, params, l_r=l_r, b1=b1, b2=b2, eps=eps)

    params = [a_tilde, z, sigma_y, sigma_a]
    args = [x, sigma_x]
    params, loss_list = train(adam, iterations, params, *args)
    a_tilde, z, sigma_y, sigma_a = params

    a_tilde = a_tilde.T
    mu, w = jnp.linalg.eig(a_tilde)
    phi = w
    b = jnp.linalg.lstsq(phi, z[0,:].T)[0]

    out = (phi @ jnp.diag(b) @ jnp.vander(mu, len(t_steps), increasing=True)).T

    # MEAN Prediction
    a_idx = index[0:-1:m//labels]
    na_idx = index[:]
    # X = self.X
    if not na_idx:
        K11 = out[a_idx, :] @ out[a_idx, :].T
        mean = jnp.linalg.inv(K11) @ x[a_idx, :]
    else:
        K11 = out[a_idx, :] @ out[a_idx, :].T
        K12 = out[a_idx, :] @ out[na_idx, :].T
        K21 = K12.T
        mean = K21 @ jnp.linalg.inv(K11 + jnp.eye(*K11.shape) * sigma_y) @ x[a_idx, :]

    # Visualize Data
    fig = plt.figure()
    grid = fig.add_gridspec(4, 1)
    ax11 = fig.add_subplot(grid[0, 0])
    ax11.set_title(f"Target Data")
    ax11.set_xlabel("Time")
    ax11.set_ylabel("$\Theta$")

    # ax11.plot(t_steps, out.T, '--')
    plt.gca().set_prop_cycle(None)
    ax11.plot(t_steps, data.transform[0].T)

    ax12 = fig.add_subplot(grid[1, 0])
    ax12.set_title(f"Reconstructed Data")
    ax12.set_xlabel("Time")
    ax12.set_ylabel("$\Theta$")

    # ax11.plot(t_steps, out.T, '--')
    plt.gca().set_prop_cycle(None)
    ax12.plot(t_steps, mean)

    ax13 = fig.add_subplot(grid[2, 0])
    ax13.set_title(f"Latent_dim")
    ax13.set_xlabel("Time")
    ax13.set_ylabel("$\Theta$")

    ax13.plot(t_steps, out, '--')

    ax14 = fig.add_subplot(grid[3, 0])
    ax14.set_title("Log_Likelihood")
    ax14.set_xlabel("Sample")
    ax14.set_ylabel("$log_likelihood value$")
    ax14.plot(jnp.arange(len(loss_list)), loss_list)
    plt.show()


if __name__ == '__main__':
    demo()