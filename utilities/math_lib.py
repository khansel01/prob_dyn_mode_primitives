import jax.numpy as jnp

from jax import jit, lax


@jit
def inv(x):
    return jnp.linalg.lstsq(x, jnp.eye(x.shape[0]), rcond=-1)[0]


@jit
def pos_def(x: jnp.ndarray) -> jnp.ndarray:
    """
    Calculate closest positive-definite symmetric NxN Matrix
    :param x: NxN Matrix
    :return: NxN Matrix
    """

    def closest_matrix(b):
        out = (b + b.conj().T) / 2
        eig_val, eig_vec = jnp.linalg.eig(out)
        out = eig_vec @ jnp.diag(jnp.maximum(eig_val, 1e-5)) @ eig_vec.conj().T
        return out.astype(x.dtype)

    return lax.cond(jnp.all(jnp.linalg.eigvals(x) > 0) & jnp.all(jnp.isclose(x, x.conj().T)),
                    lambda b: b.astype(x.dtype), closest_matrix, x)