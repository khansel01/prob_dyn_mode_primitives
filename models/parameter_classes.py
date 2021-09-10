import jax
from typing import Union, Any

from jax import lax, jit, vmap
from utils.utils_general import *
from jax.tree_util import register_pytree_node_class

# ---------------------------------------------------------------------------------------------------------------------
# -------------------- Dictronary Like Classes Storing Parameters and Compatible with JAX's JIT -----------------------
# ---------------------------------------------------------------------------------------------------------------------


@register_pytree_node_class
class ParamClass(dict):
    """ Defines a general dictionary-like structure of the parameters. This class inherits from dict and adds the
    getter, setter and delete methods for the attributes. Additonally, a tree_flatten and tree_unflatten methods are
    added
    """

    def __new__(cls: Any, *args: Any, **kwargs: Any) -> Any:

        return super().__new__(cls, *args, **kwargs)

    def __init__(self, *args, **kwargs) -> None:

        super().__init__(*args, **kwargs)

    def __getattr__(self, key: str) -> jnp.ndarray:

        if key in self:
            return self[key]
        else:
            raise AttributeError(f"No such attribute: {key}")

    def __setattr__(self, key: str, value: Union[int, jnp.ndarray]) -> None:

        self[key] = value

        return None

    def __delattr__(self, key: str) -> None:

        if key in self:
            del self[key]
        else:
            raise AttributeError(f"No such attribute: {key}")

    def tree_flatten(self):

        return jax.tree_flatten(dict(self))

    @classmethod
    def tree_unflatten(cls, aux_data, children):

        return cls(jax.tree_unflatten(aux_data, children))


@register_pytree_node_class
class DistParamClass(ParamClass):
    """ Defines a general dictionary-like structure of the parameters. This class inherits from ParamClass and adds
    property methods calculating the precision values based on the ratio of the shape and rate parameters of the
    respective gamma distribution.
    """

    def __init__(self, *args, **kwargs) -> None:
        self.alpha_y = 0.
        self.beta_y = 0.

        self.alpha_0 = jnp.zeros(1)
        self.beta_0 = jnp.zeros(1)

        self.alpha_x = 0.
        self.beta_x = 0.

        self.alpha_a = jnp.zeros((1, 1))
        self.beta_a = jnp.zeros((1, 1))

        super().__init__(*args, **kwargs)

    @property
    def lambda_y(self) -> float:

        return lax.cond(self.beta_y == 0, lambda operand: operand[0] / 1e-16, lambda operand: operand[0] / operand[1],
                        (self.alpha_y, self.beta_y))

    @property
    def lambda_0(self) -> jnp.ndarray:

        def f_scan(_, operand_in: List) -> tuple:
            return None, lax.cond(operand_in[1] == 0, lambda operand: operand[0] / 1e-16,
                                  lambda operand: operand[0] / operand[1], operand_in)

        return lax.scan(f_scan, None, [self.alpha_0, self.beta_0])[1]

    @property
    def lambda_x(self) -> float:

        return lax.cond(self.beta_x == 0, lambda operand: operand[0] / 1e-16, lambda operand: operand[0] / operand[1],
                        (self.alpha_x, self.beta_x))

    @property
    def lambda_a(self) -> jnp.ndarray:

        def f_scan(_, operand_in: List) -> tuple:
            return None, lax.cond(operand_in[1] == 0, lambda operand: operand[0] / 1e-16,
                                  lambda operand: operand[0] / operand[1], operand_in)

        return vmap(lambda alpha, beta: lax.scan(f_scan, None, [alpha, beta])[1])(self.alpha_a, self.beta_a)
