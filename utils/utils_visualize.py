import os
import matplotlib.pyplot as plt
import jax.numpy as jnp

from typing import Optional, Tuple, List
from mpl_toolkits.mplot3d import Axes3D

# ---------------------------------------------------------------------------------------------------------------------
# ------------------------------------------------ Visualization Functions --------------------------------------------
# ---------------------------------------------------------------------------------------------------------------------


def plot_grid(data: List, gridsize: Optional[Tuple[int, int]] = (1, 1), figsize: Optional[Tuple[int, int]] = (9, 6),
              save: Optional[bool] = False, save_loc: Optional[str] = None, **kwargs) -> None:
    """ Visualize the data in a grid plot using matplotlib. This function utilizes fonts and other stuff from latex.
    So latex should be installed on the system.

    :param data: A list containing the data jnp.ndarray's. All arrays should have the same number of dimensions.
    :param gridsize: Optional tuple defining the grid size (rows and columns) of the figure. If not specified '(1, 1)'
                    is taken.
    :param figsize: Optional tuple to adjust the figure size. If not specified '(9, 6)' is taken.
    :param save: Optional bool, describes whether the figure is saved. If not specified 'False' is used.
    :param save_loc: Optional str, describes where the figure should be saved. If not specified 'None' is used.
    :param kwargs:
        suptitle (str): Title of the figure. If not used it will be None.
        subtitle (str): Titles of the sub figures.If not used it will be None.
        xlabel (str): Label of the x-axis. If not used it will be 'x'.
        ylabel (str): Label of the y-axis. If not used it will be 'y'.
        color (List[str, ...]): Color of each d in data stored as string in a list.
                                If not given for each subplot blue is used.
        alpha (List[floats, ...]): Alpha values of each d in data stored as float in a list.
                                   If not given for each subplot '1.' is used.
        ls (List[int, ...]): Linestyles of each d in data stored as string in a list.
                             If not given for each subplot 'solid' is taken.
        lw (List[floats, ...]): Linewidth of each d in data stored as float in a list.
                                If not given for each subplot '1.' is taken.
        label (List[str, ...]): Label of each d in data stored as string in a list.
                                If not given for each subplot 'data' is taken.
        fill_between (List[bool, ...]): Label if for one given data the area between the max() and min() values should
                                        be visualized. If not given 'False' is taken.
        loc (List[str, ...]): Location of legend(). If not given 'best' is used.
    :return:
    """

    # --------------------------------------------------- Init -----------------------------------------------------

    suptitle = kwargs.get('suptitle', None)

    subtitle = kwargs.get('subtitle', None)

    xlabel = kwargs.get('xlabel', '$x$')

    ylabel = kwargs.get('ylabel', '$y$')

    alpha = kwargs.get("alpha", [1] * len(data))

    ls = kwargs.get("ls", ["solid"] * len(data))

    lw = kwargs.get('lw', [1.] * len(data))

    color = kwargs.get('color', ["blue"] * len(data))

    label = kwargs.get('label', ["data"] * len(data))

    fill_between = kwargs.get('fill_between', [False] * len(data))

    loc = kwargs.get('loc', ['best'] * gridsize[0] * gridsize[1])

    # --------------------------------------------- Init Latex Font --------------------------------------------------

    plt.rc('text', usetex=True)

    font = {'family': 'monospace'}

    plt.rc('font', **font)

    plt.rcParams['text.latex.preamble'] = r"\usepackage{bm} \usepackage{amsmath}"

    # ------------------------------------------------ Init Figure----------------------------------------------------

    fig = plt.figure(figsize=figsize)

    grid = plt.GridSpec(gridsize[0], gridsize[1])

    # ------------------------------------------------ Visualize -----------------------------------------------------

    for idx, g_idx in enumerate(grid):

        if idx == data[0].shape[2]:
            break

        _ax = fig.add_subplot(g_idx)

        if subtitle:
            _ax.set_title(f"{subtitle} ${idx}$", fontweight='bold')

        _ax.set_xlabel(f"{xlabel}", fontweight='bold')
        _ax.set_ylabel(f"{ylabel}", fontweight='bold')

        _ax.xaxis.set_tick_params(labelsize='large')
        _ax.yaxis.set_tick_params(labelsize='large')
        for axis in ['top', 'bottom', 'left', 'right']:
            _ax.spines[axis].set_linewidth(2)

        for d_idx, _data in enumerate(data):

            if (_data.shape[0] >= 0) & fill_between[d_idx]:

                _ax.fill_between(jnp.arange(jnp.real(_data.shape[1])), jnp.min(jnp.real(_data[:, :, idx]), axis=0),
                                 jnp.max(jnp.real(_data[:, :, idx]), axis=0), color=color[d_idx],
                                 alpha=alpha[d_idx], label=label[d_idx])

            else:

                for s_idx, _sample in enumerate(_data):
                    if s_idx == 0:
                        _label = label[d_idx]
                    else:
                        _label = None
                    _ax.plot(jnp.arange(jnp.real(_sample.shape[0])), jnp.real(_sample[:, idx]), color=color[d_idx],
                             linestyle=ls[d_idx], lw=lw[d_idx], alpha=alpha[d_idx], label=_label)

            if loc[idx]:
                _ax.legend(loc=loc[idx], prop={'size': 10})

            _ax.grid()

    if suptitle:
        fig.suptitle(suptitle)

    fig.tight_layout()

    grid.tight_layout(fig)

    # -------------------------------------------------- Save -------------------------------------------------------

    if save:
        _save_fig(figure=fig, path=save_loc, name=suptitle)
    else:
        fig.show()

    return None


def plot_3d(data: List, gridsize: Optional[Tuple[int, int]] = (1, 1), figsize: Optional[Tuple[int, int]] = (9, 6),
            save: Optional[bool] = False, save_loc: Optional[str] = None, **kwargs) -> None:
    """ Visualize the data as 3d plot in a grid plot using matplotlib. This function utilizes fonts and other stuff
    from latex. So latex should be installed on the system.

    :param data: A list containing the data jnp.ndarray's. All arrays should have the same number of dimensions.
    :param gridsize: Optional tuple defining the grid size (rows and columns) of the figure. If not specified '(1, 1)'
                    is taken.
    :param figsize: Optional tuple to adjust the figure size. If not specified '(9, 6)' is taken.
    :param save: Optional bool, describes whether the figure is saved. If not specified 'False' is used.
    :param save_loc: Optional str, describes where the figure should be saved. If not specified 'None' is used.
    :param kwargs:
        suptitle (str): Title of the figure. If not used it will be None.
        subtitle (str): Titles of the sub figures.If not used it will be None.
        xlabel (str): Label of the x-axis. If not used it will be 'x'.
        ylabel (str): Label of the y-axis. If not used it will be 'y'.
        zlabel (str): Label of the z-axis. If not used it will be 'z'.
        color (List[str, ...]): Color of each d in data stored as string in a list.
                                If not given for each subplot blue is used.
        alpha (List[floats, ...]): Alpha values of each d in data stored as float in a list.
                                   If not given for each subplot '1.' is used.
        ls (List[int, ...]): Linestyles of each d in data stored as string in a list.
                             If not given for each subplot 'solid' is taken.
        lw (List[floats, ...]): Linewidth of each d in data stored as float in a list.
                                If not given for each subplot '1.' is taken.
        label (List[str, ...]): Label of each d in data stored as string in a list.
                                If not given for each subplot 'data' is taken.
        fill_between (List[bool, ...]): Label if for one given data the area between the max() and min() values should
                                        be visualized. If not given 'False' is taken.
        loc (List[str, ...]): Location of legend(). If not given 'best' is used.
        elev (List[float, ...]): Stores the elevation angle in the z plan. If not given 'None' is used.
        azim (List[float, ...]):  Stores the azimuth angle in the x,y plan. If not given 'None' is used.
    :return:
    """

    # --------------------------------------------------- Init -----------------------------------------------------

    suptitle = kwargs.get('suptitle', None)

    subtitle = kwargs.get('subtitle', None)

    xlabel = kwargs.get('xlabel', '$x$')

    ylabel = kwargs.get('ylabel', '$y$')

    zlabel = kwargs.get('zlabel', '$z$')

    alpha = kwargs.get("alpha", [1] * len(data))

    ls = kwargs.get("ls", ["solid"] * len(data))

    lw = kwargs.get('lw', [1.] * len(data))

    color = kwargs.get('color', ["blue"] * len(data))

    label = kwargs.get('label', ["data"] * len(data))

    loc = kwargs.get('loc', ['best'] * gridsize[0] * gridsize[1])

    zorder = kwargs.get('zorder', jnp.arange(len(data)))

    elev = kwargs.get('elev', [None] * gridsize[0] * gridsize[1])

    azim = kwargs.get('azim', [None] * gridsize[0] * gridsize[1])

    # --------------------------------------------- Init Latex Font --------------------------------------------------

    plt.rc('text', usetex=True)

    font = {'family': 'monospace'}

    plt.rc('font', **font)

    plt.rcParams['text.latex.preamble'] = r"\usepackage{bm} \usepackage{amsmath}"

    # ------------------------------------------------ Init Figure----------------------------------------------------

    fig = plt.figure(figsize=figsize)

    grid = plt.GridSpec(gridsize[0], gridsize[1])

    # ------------------------------------------------ Visualize -----------------------------------------------------

    for idx, g_idx in enumerate(grid):

        if idx == data[0].shape[2]:
            break

        _ax = fig.add_subplot(g_idx, projection='3d')

        if subtitle:
            _ax.set_title(f"{subtitle} ${idx}$", fontweight='bold')

        _ax.set_xlabel(f"{xlabel}", fontweight='bold')
        _ax.set_ylabel(f"{ylabel}", fontweight='bold')
        _ax.set_zlabel(f"{zlabel}", fontweight='bold')

        _ax.xaxis.set_tick_params(labelsize='large')
        _ax.yaxis.set_tick_params(labelsize='large')
        _ax.zaxis.set_tick_params(labelsize='large')
        for axis in ['top', 'bottom', 'left', 'right']:
            _ax.spines[axis].set_linewidth(2)

        for d_idx, _data in enumerate(data):

            # avoid overlapping if alpha value is used
            _traj =jnp.concatenate(jnp.concatenate((_data, jnp.nan * jnp.ones((_data.shape[0], 1, _data.shape[2]))),
                                                   axis=1))

            _ax.plot(jnp.real(_traj.T[0]), jnp.real(_traj.T[1]), jnp.real(_traj.T[2]),
                     color=color[d_idx], linestyle=ls[d_idx], lw=lw[d_idx], alpha=alpha[d_idx], label=label[d_idx],
                     zorder=zorder[d_idx])

            if loc[idx]:
                _ax.legend(loc=loc[idx], prop={'size': 10})

            _ax.grid()
            _ax.xaxis.pane.fill = False
            _ax.yaxis.pane.fill = False
            _ax.zaxis.pane.fill = False

            _ax.xaxis._axinfo['tick']['inward_factor'] = 0
            _ax.xaxis._axinfo['tick']['outward_factor'] = 0.4
            _ax.yaxis._axinfo['tick']['inward_factor'] = 0
            _ax.yaxis._axinfo['tick']['outward_factor'] = 0.4
            _ax.zaxis._axinfo['tick']['inward_factor'] = 0
            _ax.zaxis._axinfo['tick']['outward_factor'] = 0.4
            _ax.zaxis._axinfo['tick']['outward_factor'] = 0.4

        _ax.view_init(elev=elev[idx], azim=azim[idx])

    if suptitle:
        fig.suptitle(suptitle)

    fig.tight_layout()

    grid.tight_layout(fig)

    # -------------------------------------------------- Save -------------------------------------------------------

    if save:
        _save_fig(figure=fig, path=save_loc, name=suptitle)
    else:
        fig.show()

    return None


def _save_fig(figure: plt.Figure, path: str, name: str) -> None:
    """ Save the figure as a PDF file.

    :param figure: A given matplotlib Figure.
    :param path: A str specifying the path where to save the figure.
    :param name: A str corresponding to the name of the figure.
    :return:
    """

    assert name, ValueError(f"Name of figure not properly defined.(given name: {name})")

    assert path, ValueError(f"A path should be given where to save the figure.(given path: {path})")

    assert os.path.exists(path), FileNotFoundError(f"The specified path does not exists. (given path: {path})")

    name_refactored = '_'.join(name.lower().split())

    paht_final = f'{path}/{name_refactored}.pdf'

    figure.savefig(paht_final, format='pdf', dpi=1200)

    print(f'\t Figure saved in {paht_final}.')

    return None
