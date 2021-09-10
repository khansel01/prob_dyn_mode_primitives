import numpy as np
import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D
from typing import List
from utils.utils_general import load_recordings, preprocessing


def demo() -> None:
    """ This demo shows an example of how to load and preprocess recorded data

    :return: None
    """

    # ---------------------------------------- load and preprocess data ----------------------------------------------

    data = load_recordings(path='../../data/recordings/letters_21_08/c_letter')

    pp_data = preprocessing(t_steps=1000, data=data)

    # ----------------------------------------------- visualize ------------------------------------------------------

    f_x = plt.figure()

    grid_x = f_x.add_gridspec(1, 2)

    ax_x = [f_x.add_subplot(grid_x[0, 0], projection='3d'), f_x.add_subplot(grid_x[0, 1], projection='3d')]

    for _ax in ax_x:
        _ax.set_xlabel('x')
        _ax.set_ylabel('y')
        _ax.set_zlabel('z')

    f_q, ax_q = plt.subplots(4, 2)

    t = np.arange(0, 1000)

    t = t / t[-1]

    def sub_plot(_dict: dict, _pp_dict: dict, color_idx: int=1) -> None:
            for idx in range(_dict['q'].shape[1]):
                ax_q[idx, 0].plot(np.arange(_dict['q'].shape[0]), _dict['q'][:, idx], color=f'C{color_idx}')
                ax_q[idx, 0].set_xlabel('time')
                ax_q[idx, 0].set_ylabel('position')
                ax_q[idx, 0].set_title(f'Joint {idx}')

                ax_q[idx, 1].plot(np.arange(_pp_dict['q'].shape[0]), _pp_dict['q'][:, idx], color=f'C{color_idx}')
                ax_q[idx, 1].set_xlabel('time')
                ax_q[idx, 1].set_ylabel('position')
                ax_q[idx, 1].set_title(f'Joint {idx}')
            ax_x[0].plot(_dict["x"][:, 0], _dict["x"][:, 1], _dict['x'][:, 2],
                         color=f'C{color_idx}', alpha=0.5, linewidth=2.0)
            ax_x[1].plot(_pp_dict["x"][:, 0], _pp_dict["x"][:, 1], _pp_dict["x"][:, 2],
                         color=f'C{color_idx}', alpha=0.5, linewidth=2.0)
            return None

    for idx, content in enumerate(zip(data, pp_data)):

        if isinstance(content[0], dict):

            sub_plot(*content, color_idx=idx)

        elif isinstance(content[0], List):

            for jdx, trajectory in enumerate(zip(*content)):
                sub_plot(*trajectory, color_idx=idx)

        else:

            raise ValueError("The given data does not have a valid structure.")

    plt.show()

    return None


if __name__ == '__main__':
    demo()


