import numpy as np
import jax
import jax.numpy as jnp
from scipy import linalg


def data2snapshots(data: np.ndarray, t_delay: int = 1, axis: int = 0) -> tuple:
    """
    Converts the batch containing the data of all trajectories into two snapshot matrices X0 and X1
    :param data: Data as np.ndarray of size b x n x m.
    :param t_delay: Integer to set the delay coordinates.
    :param axis: Integer, specifies along which axis the data will be concatenated
    :return: Batch containing two snapshot matrices X0 and X1
    """
    b, n, m = data.shape

    snapshots = data[:, :, :m - t_delay]
    for delay in range(1, t_delay + 1):
        snapshots = np.append(snapshots, data[:, :, delay:m + delay - t_delay], axis=1)

    return np.concatenate(snapshots[:, :-n], axis=axis), np.concatenate(snapshots[:, n:], axis=axis)


def snapshots2data(snapshots: np.array, samples: int = 1, t_delay: int = 1, axis: int = 0):
    """TODO
    Convert Snapshot matrix into a list containing all trajectories as separate np.arrays
    :param snapshots: Snapshot matrix as np. array
    :param samples: Integer how many trajectories are contained in X
    :param t_delay: Integer
    :param axis: Integer
    :return:
    """
    data = np.array(np.split(snapshots, samples, axis=axis))

    b, n, m = data.shape
    c = n // (t_delay + 1)

    return np.append(data[:, :c, :], data[:, c:, -1:].reshape(b, c, t_delay, order='F'), axis=2)


def pos_def(x: np.ndarray) -> np.ndarray:
    """
    Calculate closest positive-definite symmetric NxN Matrix
    :param x: NxN Matrix
    :return: NxN Matrix
    """
    if not np.all(linalg.eigvals(x) > 0):
        y = (x + x.conj().T) / 2
        eig_val, eig_vec = linalg.eig(y)
        y = eig_vec @ np.diag(np.clip(eig_val, a_min=0, a_max=None)) @ eig_vec.T
        return y
    return x

def sqrtm(X):
    # TODO
    Y = X.copy()
    Z = jnp.eye(len(X))

    error = 1
    error_tolerance = 1.5e-8

    flag = 1
    while error > error_tolerance:
        Y_old = Y
        Y = (Y_old + jnp.linalg.inv(Z))/2
        Z = (Z + jnp.linalg.inv(Y_old))/2
        error_matrix = abs(Y - Y_old)
        error = 0
        # detect the maximum value in the error matrix
        for i in range(len(X)):
            temp_error = max(error_matrix[i])
            if temp_error > error:
                error = temp_error

        flag = flag + 1

    print("Iteration Times: ", flag)
    return Y