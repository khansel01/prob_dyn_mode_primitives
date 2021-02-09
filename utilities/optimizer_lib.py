import jax
import jax.numpy as jnp

from jax import jit
from jax.experimental import optimizers as opt


class Adam(object):
    def __init__(self, loss: jnp.function, **kwargs):
        self.loss = loss

        self.l_r = kwargs.get("l_r", 1e-3)
        self.b1 = kwargs.get("b1", 0.9)
        self.b2 = kwargs.get("b2", 0.999)
        self.eps = kwargs.get("eps", 1e-8)

        self.opt_init, self.opt_update, self.get_params = opt.adam(self.l_r, b1=self.b1, b2=self.b2, eps=self.eps)

    @jax.partial(jit, static_argnums=(0, 3,))
    def step(self, params, opt_state, kernel_fun, *args):
        value, grads = jax.value_and_grad(self.loss)(params, kernel_fun, *args)
        opt_state = self.opt_update(0, grads, opt_state)
        return self.get_params(opt_state), opt_state, value
