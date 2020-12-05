"""
This file generates Minimum-Jerk Trajectories.

References
Paper: A Minimum-Jerk Trajectory (http://courses.shadmehrlab.org/Shortcourse/minimumjerk.pdf)
Github: https://github.com/ekorudiawan/Minimum-Jerk-Trajectory
"""

import torch as tr


class MinimumJerk(object):
    def __init__(self):
        pass

    @staticmethod
    def get_data(t_steps: tr.tensor, x_init: tr.tensor, x_final: tr.tensor,
                 trajectories: int=1, sigma: float=0.001):
        """
        Get a list containing minimum jerk trajectories
        :param t_steps: Time steps as a T dimensional torch tensor
        :param x_init: Torch tensor of size Nx1
        :param x_final: Torch tensor of size Nx1
        :param trajectories: Amount of trajectories as integer
        :param sigma: Variability in state and end point
        :return: a torch tensor of size BxNxT containing minimum jerk trajectories
        """
        t_end = t_steps[-1].item()
        n = x_init.shape[0]
        data = tr.zeros(trajectories, n * 3, t_steps.shape[0])

        for t in range(trajectories):
            _x_init = x_init + tr.randn_like(x_init)*sigma
            _x_final = x_final + tr.randn_like(x_final)*sigma

            delta_x = _x_final - _x_init

            data[t, 0:n, :] = _x_init + delta_x * (10 * (t_steps / t_end) ** 3
                                                   - 15 * (t_steps / t_end) ** 4
                                                   + 6 * (t_steps / t_steps[-1]) ** 5)

            data[t, n:2*n, :] = delta_x * (30 / t_steps[-1] * (t_steps / t_steps[-1]) ** 2
                                           - 60 / t_steps[-1] * (t_steps / t_steps[-1]) ** 3
                                           + 30 / t_steps[-1] * (t_steps / t_steps[-1]) ** 4)

            data[t, 2*n:3*n, :] = delta_x * (60 / t_steps[-1] ** 2 * (t_steps / t_steps[-1])
                                             - 180 / t_steps[-1] ** 2 * (t_steps / t_steps[-1]) ** 2
                                             + 120 / t_steps[-1] ** 2 * (t_steps / t_steps[-1]) ** 3)

        return data
