import os
import datetime
import numpy as np
import pickle
import jax.numpy as jnp

from scipy import interpolate
from typing import List, Optional, Dict
from copy import deepcopy

# ---------------------------------------------------------------------------------------------------------------------
# ------------------------------------------------ General Utility Functions ------------------------------------------
# ---------------------------------------------------------------------------------------------------------------------


def save_dict(dict_to_save: Dict, path: str, name: Optional[str] = 'data') -> None:
    """ Save a dictionary containing all the GPDMD parameters under a given path as pickle file.

    :param dict_to_save: A dist type containing all relevant parameters.
    :param path: A str specifying the path where to save the figure.
    :param name: Optional str corresponding to the name of the pickle file. If not given 'data' is used.
    :return:
    """

    assert name, ValueError(f"Name of figure not properly defined.(given name: {name})")

    assert path, ValueError(f"A path should be given where to save the figure.(given path: {path})")

    assert os.path.exists(path), FileNotFoundError(f"The specified path does not exists. (given path: {path})")

    name_refactored = '_'.join(name.lower().split())

    with open(f'{path}/{name_refactored}.pkl', 'wb') as file:
        pickle.dump(dict_to_save, file, pickle.HIGHEST_PROTOCOL)

    print(f'\t Dict saved as {path}/{name_refactored}.pkl.')

    return None


def load_dict(path: str, name: Optional[str] = 'data') -> Dict:
    """ Load a dictionary containing all relevant parameters for a GPDMD-like object.

    :param path: A str specifying the path.
    :param name: Optional str corresponding to the name of the pickle file. If not given 'data' is used.
    :return:
    """

    file_loc = f'{path}/{name}.pkl'

    assert os.path.exists(file_loc), FileNotFoundError(f"The specified path does not exists. "
                                                       f"(given path: {file_loc})")

    with open(file_loc, 'rb') as file:
        dict_to_load = pickle.load(file)

    print(f'\t Dict loaded from the path {file_loc}.')

    return dict_to_load


def create_folder(path: str) -> str:
    """ If data of the rund should be saved a new folder will be created.

    :param path: Path where the folder should be located.
    :return: A path to the new created folder.
    """

    assert path, ValueError(f"A path should be given where to save the figure.(given path: {path})")

    assert os.path.exists(path), FileNotFoundError(f"The specified path does not exists. (given path: {path})")

    path_to_folder = f'{path}/{datetime.datetime.now():%Y_%m_%d__%H_%M_%S}'

    assert not os.path.exists(path_to_folder), FileNotFoundError(f"The specified folder already exists. "
                                                                 f"(current folder: {path_to_folder}")

    os.mkdir(path_to_folder)

    print(f'\t Folder {path_to_folder} created.')

    return path_to_folder


def load_recordings(path: str) -> List:
    """ Load recorded data.

    :param path: as str. Specify the path for the file or dictionary of recorded data.
    :return: data: as list. Containing the loaded data.
    """
    assert os.path.exists(path), FileNotFoundError("The specified path does not exists.")

    data = []
    if os.path.isfile(path):
        with open(path, "rb") as f:
            data.append(pickle.load(f))
        return data
    elif os.path.isdir(path):
        for subdir, dirs, files in os.walk(path, topdown=False):
            trajs = []
            for file in sorted(files):
                with open(os.path.join(subdir, file), "rb") as f:
                    trajs.append(pickle.load(f))
            data.append(trajs)
        return data
    else:
        raise ValueError("The specified path is neither a dictionary nor a file.")


def preprocessing(t_steps: int, data: List) -> List:
    """ This function preprocesses recorded data by a normalization over the time.

    :param t_steps: as int. Corresponds to the desired time steps.
    :param data: a list containing the recorded data.
    :return: pp_data as list representing the preprocessed data.
    """
    pp_data = deepcopy(data)

    t = np.arange(t_steps) / (t_steps - 1)

    def sub_func(_dict: dict) -> dict:
        for _key, _trajectory in _dict.items():
            _t = np.arange(_trajectory.shape[0]) / (_trajectory.shape[0] - 1)
            pp_trajectory = np.zeros((t.shape[0], _trajectory.shape[1]))
            for _j in range(_trajectory.shape[1]):
                pp_trajectory[:, _j] = interpolate.interp1d(_t, _trajectory[:, _j], kind='cubic')(t)
            _dict.update({_key: pp_trajectory})
        return _dict

    for idx, content in enumerate(pp_data):
        if isinstance(content, dict):
            pp_data[idx] = sub_func(content)
        elif isinstance(content, List):
            pp_content = content.copy()
            for jdx, trajectory in enumerate(content):
                pp_content[jdx] = sub_func(trajectory)
            pp_data[idx] = pp_content
        else:
            raise ValueError("The given data does not have a valid structure.")

    return pp_data


def data2snapshots(data: jnp.ndarray, t_delay: int = 1, axis: int = 0) -> tuple:
    """ Converts the batch containing the data of all trajectories into two snapshot matrices X0 and X1

    :param data: Data as jnp.ndarray of size b x n x m.
    :param t_delay: Integer to set the delay coordinates.
    :param axis: Integer, specifies along which axis the data will be concatenated
    :return: Batch containing two snapshot matrices X0 and X1
    """
    b, n, m = data.shape

    snapshots = data[:, :, :m - t_delay]
    for delay in range(1, t_delay + 1):
        snapshots = jnp.append(snapshots, data[:, :, delay:m + delay - t_delay], axis=1)

    return jnp.concatenate(snapshots[:, :-n], axis=axis), jnp.concatenate(snapshots[:, n:], axis=axis)


def snapshots2data(snapshots: jnp.array, samples: int = 1, t_delay: int = 1, axis: int = 0) -> jnp.ndarray:
    """ Convert Snapshot matrix into a list containing all trajectories as separate jnp.arrays

    :param snapshots: Snapshot matrix as jnp. array
    :param samples: Integer how many trajectories are contained in X
    :param t_delay: Integer
    :param axis: Integer
    :return:
    """
    data = jnp.array(jnp.split(snapshots, samples, axis=axis))

    b, n, m = data.shape
    c = n // (t_delay + 1)

    return jnp.append(data[:, :c, :], data[:, c:, -1:].reshape(b, c, t_delay, order='F'), axis=2)

# ---------------------------------------------------------------------------------------------------------------------
# -------------------------------------------- General Utility Classes ------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------------


class Standardization(object):
    def __init__(self, data: jnp.ndarray):
        """ Standardization of given data.

        :param data: is represented as NxT JAX.ndarry,
        where N and T correspond to the number of dimensions and time, respectively.
        """
        self.data = data

    def __call__(self):
        return self.forward()

    @property
    def mu(self):
        return jnp.sum(self.data, axis=1, keepdims=True) / self.data.shape[1]

    @property
    def sigma(self):
        return jnp.diag((self.data - self.mu) @ (self.data - self.mu).T / self.data.shape[1])[:, None]

    def forward(self):
        return jnp.reciprocal(self.sigma) * (self.data - self.mu)

    def backward(self):
        return self.sigma * self.forward() + self.mu


class Whitening(object):
    def __init__(self, data: jnp.ndarray):
        """ Whitening of given data.

        :param data: is represented as NxT JAX.ndarry,
        where N and T correspond to the number of dimensions and time, respectively.
        """
        self.data = data

    def __call__(self):
        return self.forward()

    @property
    def mu(self):
        return jnp.sum(self.data, axis=1, keepdims=True) / self.data.shape[1]

    @property
    def sigma(self):
        return (self.data - self.mu) @ (self.data - self.mu).T / self.data.shape[1]

    @property
    def std_u(self):
        _lambda, u = jnp.linalg.eigh(self.sigma)
        return jnp.sqrt(_lambda) * jnp.eye(_lambda.shape[0]), u

    def forward(self):
        std, u = self.std_u
        return jnp.linalg.lstsq(std, jnp.eye(std.shape[0]))[0] @ u.T @ (self.data - self.mu)

    def backward(self):
        std, u = self.std_u
        return u @ std @ self.forward() + self.mu
