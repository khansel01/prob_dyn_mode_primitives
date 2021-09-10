import argparse
import jax
import itertools

from demos import *

# --------------------------------------------------------------------------------------------------------------------
# ------------------------------------------------ Main Example of GP-DMD --------------------------------------------
# --------------------------------------------------------------------------------------------------------------------


def main():
    """
    This demo shows the main example of Gaussian Process Dynamic Mode Decomposition on an eight-shape dataset
    TODO Play a little bit around and learn Circle Shape, and in the joint space I, C, R, and A.
    :return: None
    """

    parser = argparse.ArgumentParser(description="Eight Shape")

    parser.add_argument("--device", default="cpu", type=str,
                        help="Specify the device on which Jax works. Possible devices are cpu or gpu (if available).")
    parser.add_argument("--x64", default=True, type=bool,
                        help="Defines the float type used by Jax.")

    parser.add_argument("--seed", default=11, type=int,
                        help="Defines the seed of the pseudo random number generator.")
    parser.add_argument("--variability", default=0.05, type=float,
                        help="Set the variability in the given demonstrations in the data generator as variance "
                             "parameter.")
    parser.add_argument("--number_demonstrations", default=20, type=int,
                        help="Set given number of demonstrations if a data generator is used.")

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
                        help='Shape parameter of the gamma distribution corresponding to the variance of the '
                             'emission model')
    parser.add_argument('--beta_y', default=1e-4, type=float,
                        help='Rate parameter of the gamma distribution corresponding to the variance of the '
                             'emission model')

    parser.add_argument('--kappa_0', default=1e-16, type=float,
                        help='Amplification parameter of the a normal gamma distribution amplifying the variance '
                             'of the initial state distribution.')
    parser.add_argument('--alpha_0', default=1. + 1e-16, type=float,
                        help='Shape parameter of the normal gamma distribution corresponding to the initial state '
                             'distribution')
    parser.add_argument('--beta_0', default=1e-4, type=float,
                        help='Rate parameter of the normal gamma distribution corresponding to the initial state '
                             'distribution')

    parser.add_argument('--alpha_x', default=1. + 1e-16, type=float,
                        help='Shape parameter of the gamma distribution corresponding to the variance of the '
                             'transition model')
    parser.add_argument('--beta_x', default=1e-4, type=float,
                        help='Rate parameter of the gamma distribution corresponding to the variance of the transition '
                             'model')

    parser.add_argument('--kappa_a', default=1e-16, type=float,
                        help='Amplification parameter of the a normal gamma distribution amplifying the variance '
                             'of the distribution of the hierarchical model.')
    parser.add_argument('--alpha_a', default=1. + 1e-16, type=float,
                        help='Shape parameter of the normal gamma distribution corresponding to the hierarchical model')
    parser.add_argument('--beta_a', default=1e16, type=float,
                        help='Rate parameter of the normal gamma distribution corresponding to the hierarchical model')

    parser.add_argument("--num_samples", default=1000, type=int,
                        help="Number of samples of the learned GP-DMD.")

    parser.add_argument("--save_params", default=False, type=bool,
                        help="Specifies whether the trained model is saved.")
    parser.add_argument("--save_fig", default=True, type=bool,
                        help="Specifies whether the figures are saved.")
    parser.add_argument("--save_loc", default=f'/home/kay/projects/gp_dmd/runs', type=str,
                        help="Specifies the path to the folder where the files should be saved.")

    args = parser.parse_args()

    jax.config.update('jax_platform_name', args.device)

    jax.config.update('jax_enable_x64', args.x64)

    list_vals = [1e-1, 1e-2, 1e-3, 1e-4]

    storage = []

    for count, elem in enumerate(itertools.product(list_vals, list_vals, list_vals[2:])):

        args.beta_y, args.beta_x, args.beta_0 = elem

        print(f'\t Start with beta_a = {elem[0]:.1e} \t beta_y = {elem[1]:.1e} \t beta_0 = {elem[2]:.1e}')

        mse, std = circle_shape_example(args)

        storage.append([count, args.beta_y, args.beta_x, args.beta_0, mse, std])

    for elem in storage:

        print(f'-{elem[0]:04d}) \t beta_a = {elem[1]:.1e} \t beta_y = {elem[2]:.1e} \t beta_0 = {elem[3]:.1e} '
              f'with mse: {elem[4]:.5e} and std: {elem[5]:.5e}')


if __name__ == '__main__':
    main()
