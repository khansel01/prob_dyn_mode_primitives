"""Several utility methods """

import jax.numpy as jnp

from jax import jit


def data2snapshots(data: jnp.ndarray, t_delay: int = 1, axis: int = 0) -> tuple:
    """
    Converts the batch containing the data of all trajectories into two snapshot matrices X0 and X1
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


def snapshots2data(snapshots: jnp.array, samples: int = 1, t_delay: int = 1, axis: int = 0):
    """TODO
    Convert Snapshot matrix into a list containing all trajectories as separate jnp.arrays
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


def pos_def(x: jnp.ndarray) -> jnp.ndarray:
    """
    Calculate closest positive-definite symmetric NxN Matrix
    :param x: NxN Matrix
    :return: NxN Matrix
    """
    if not jnp.all(jnp.linalg.eigvals(x) > 0):
        y = (x + x.conj().T) / 2
        eig_val, eig_vec = jnp.linalg.eig(y)
        y = eig_vec @ jnp.diag(jnp.clip(eig_val, a_min=0, a_max=None)) @ eig_vec.T
        return y
    return x
