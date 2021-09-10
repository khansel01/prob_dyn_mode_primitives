import argparse
import jax
import jax.numpy as jnp

from jax import vmap
from typing import Tuple
from models.gp_dmd import GPDMD
from utils.prng_handler import PRNGHandler
from utils.utils_general import Whitening, create_folder, save_dict, load_recordings, preprocessing
from utils.utils_kernels import ard_kernel
from utils.utils_visualize import plot_grid, plot_3d


parser = argparse.ArgumentParser(description="R Letter")

parser.add_argument("--device", default="cpu", type=str,
                    help="Specify the device on which Jax works. Possible devices are cpu or gpu (if available).")
parser.add_argument("--x64", default=True, type=bool,
                    help="Defines the float type used by Jax.")

parser.add_argument("--seed", default=11, type=int,
                    help="Defines the seed of the pseudo random number generator.")

parser.add_argument("--latent_dim", default=4, type=int,
                    help="Dimensionality of the latent space.")
parser.add_argument("--i_number", default=50, type=int,
                    help="Specifies the number of inducing pairs.")

parser.add_argument("--lr_init", default=1e-1, type=float,
                    help="Specifies the learning rate of the MAP training to initialize GP-DMD.")
parser.add_argument("--iterations_init", default=2500, type=int,
                    help="Specifies the number of iterations of the MAP estimate.")
parser.add_argument("--lr_main", default=None, type=float,
                    help="Specifies the learning rate of the fully Bayesian training.")
parser.add_argument("--iterations_main", default=1000, type=int,
                    help="Specifies the number of iterations of the fully Bayesian training.")

parser.add_argument("--gamma", default=1., type=float,
                    help="Kernel Parameter gamma.")
parser.add_argument("--theta", default=1., type=float,
                    help="Kernel Parameter theta.")

parser.add_argument('--alpha_y', default=1. + 1e-16, type=float,
                    help='Shape parameter of the gamma distribution corresponding to the variance of the emission '
                         'model')
parser.add_argument('--beta_y', default=1e-2, type=float,
                    help='Rate parameter of the gamma distribution corresponding to the variance of the emission '
                         'model')

parser.add_argument('--kappa_0', default=1e-16, type=float,
                    help='Amplification parameter of the a normal gamma distribution amplifying the variance  of the '
                        'initial state distribution.')
parser.add_argument('--alpha_0', default=1. + 1e-16, type=float,
                    help='Shape parameter of the normal gamma distribution corresponding to the initial state '
                        'distribution')
parser.add_argument('--beta_0', default=1e-2, type=float,
                    help='Rate parameter of the normal gamma distribution corresponding to the initial state '
                        'distribution')

parser.add_argument('--alpha_x', default=1. + 1e-16, type=float,
                    help='Shape parameter of the gamma distribution corresponding to the variance of the transition '
                        'model')
parser.add_argument('--beta_x', default=1e-2, type=float,
                    help='Rate parameter of the gamma distribution corresponding to the variance of the transition '
                        'model')

parser.add_argument('--kappa_a', default=1e-16, type=float,
                    help='Amplification parameter of the a normal gamma distribution amplifying the variance  of the '
                        'distribution of the hierarchical model.')
parser.add_argument('--alpha_a', default=1. + 1e-16, type=float,
                    help='Shape parameter of the normal gamma distribution corresponding to the hierarchical model')
parser.add_argument('--beta_a', default=1e16, type=float,
                    help='Rate parameter of the normal gamma distribution corresponding to the hierarchical model')

parser.add_argument("--num_samples", default=1000, type=int,
                    help="Number of samples of the learned GP-DMD.")

parser.add_argument("--save_params", default=False, type=bool,
                    help="Specifies whether the trained model is saved.")
parser.add_argument("--save_fig", default=False, type=bool,
                    help="Specifies whether the figures are saved.")
parser.add_argument("--save_loc", default=f'/home/kay/projects/gp_dmd/runs', type=str,
                    help="Specifies the path to the folder where the files should be saved.")

# --------------------------------------------------------------------------------------------------------------------
# --------------------------------------- Example of GP-DMD on the R Letter --------==--------------------------------
# --------------------------------------------------------------------------------------------------------------------


def r_letter_example(args: argparse.Namespace) -> Tuple:
    """
    Example on the R Letter
    :return: None
    """

    prng_handler = PRNGHandler(seed=args.seed)

    t_steps = jnp.arange(0, 1, 0.01)

    # -------------------------------------------- generate data -----------------------------------------------------

    data = preprocessing(100, load_recordings('data/recordings/letters_21_08/r_letter'))

    data = jnp.array([trajectory["q"] for trajectory in data[0]])[5:]

    y = jnp.concatenate(data, axis=0)

    standarizer = Whitening(y.reshape(1, -1, order='F'))

    y = jnp.float64(standarizer().T.reshape(*y.shape, order='F'))

    y = y.reshape(data.shape)

    # -------------------------------------------- Training process --------------------------------------------------

    _kwargs = {'iterations_init': args.iterations_init, 'iterations_main': args.iterations_main,
               'lr_init': args.lr_init, 'lr_main': args.lr_main,
               'kernel_fun': ard_kernel, 'gamma': args.gamma, 'theta': args.theta,
               "alpha_y": args.alpha_y, "beta_y": args.beta_y,
               "kappa_0": args.kappa_0, "alpha_0": args.alpha_0, "beta_0": args.beta_0,
               "alpha_x": args.alpha_x, "beta_x": args.beta_x,
               "kappa_a": args.kappa_a, "alpha_a": args.alpha_a, "beta_a": args.beta_a}

    gp_dmd = GPDMD(latent_dim=args.latent_dim, i_number=args.i_number, prng_handler=prng_handler, **_kwargs)

    gp_dmd.fit(y)

    # ----------------------------------------- Prediction Latent Space ----------------------------------------------

    x, mu_y, _ = gp_dmd.predict(t_steps)

    # ---------------------------------------------- Sampling --------------------------------------------------------

    x_samples, mean_samples = gp_dmd.sampling(t_steps, num_samples=args.num_samples)

    # ---------------------------------------- Save and Visualization ------------------------------------------------

    if args.save_params:
        save_loc = create_folder(args.save_loc)
        save_dict(gp_dmd.as_dict(), save_loc)
    elif args.save_fig:
        save_loc = create_folder(args.save_loc)
    else:
        save_loc = args.save_loc

    _kwargs = {'suptitle': 'R Letter in Joint Space', 'subtitle': '$\mathrm{Joint}$',
               'xlabel': '$\mathrm{Time}$', 'ylabel': '$\Theta$', 'color': ["red", "blue", "blue"],
               'label': ["$\mathrm{Data}$", "$\mathrm{Mean}$", "$\mathrm{Variance}$"],
               'alpha': [1., 1., 0.3], 'ls': ["dashed", "solid", None], 'loc': ['best', None, None, None],
               'lw': [0.5, 1., None], 'fill_between': [False, False, True]}
    plot_grid([y, mu_y[None, :], mean_samples], gridsize=(2, 2), save=args.save_fig, save_loc=save_loc, **_kwargs)

    _kwargs = {'suptitle': r'R Letter in Latent Space', 'subtitle': '$\mathrm{Latent~dim}$',
               'xlabel': '$\mathrm{Time}$', 'ylabel': '$x$', 'color': ['red', "blue", "blue"],
               'label': ["$\mathrm{True}$", "$\mathrm{Mean}$", "$\mathrm{Variance}$"],
               'alpha': [1., 1., 0.3], 'ls': ["dashed", "solid", None],
               'loc': ['best', None, None, None, None, None],
               'lw': [0.5, 1., None], 'fill_between': [False, False, True]}
    plot_grid([gp_dmd.opt_params.x, x[None, :], x_samples], gridsize=(2, 2), save=args.save_fig,
              save_loc=save_loc, **_kwargs)

    # ---------------------------------------------- Prints Error ----------------------------------------------------

    y_mu = jnp.sum(y, axis=0) / y.shape[0]

    mse_mu = jnp.trace((mu_y - y_mu) @ (mu_y - y_mu).T) / mu_y.shape[0]

    mse_samples = vmap(jnp.trace)((mean_samples - y_mu) @ (mean_samples - y_mu).transpose((0, 2, 1))) / mu_y.shape[0]

    delta_mse = mse_samples - mse_mu

    mse_std = jnp.sqrt(jnp.sum(delta_mse ** 2) / delta_mse.shape[0])

    return jnp.abs(mse_mu), jnp.abs(mse_std)


if __name__ == '__main__':

    args = parser.parse_args()

    jax.config.update('jax_platform_name', args.device)

    jax.config.update('jax_enable_x64', args.x64)

    mse, std = r_letter_example(args)

    print(f'\t The Mean Squared Error: {mse:.5e}')

    print(f'\t The Standard Deviation of the Mean Squared Error: {std:5e}')
