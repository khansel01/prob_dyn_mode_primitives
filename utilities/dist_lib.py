import jax
import jax.numpy as jnp

from jax import jit
from jax.scipy.special import gammaln, digamma
from utilities.math_lib import inv, pos_def


class Gamma:
    @staticmethod
    @jit
    def std_to_nat(std_param):
        nat_param1 = std_param[0] - 1
        nat_param2 = - std_param[1]
        return [nat_param1, nat_param2]

    @staticmethod
    @jit
    def nat_to_std(nat_param):
        alpha = nat_param[0] + 1
        beta = - nat_param[1]
        return [alpha, beta]

    @staticmethod
    @jit
    def log_likelihood(std_param, x):
        ll_value = (std_param[0] - 1.) * jnp.log(x) - std_param[1] * x

        ll_value -= gammaln(std_param[0]) - std_param[0] * jnp.log(std_param[1])

        return ll_value


class Gaussian:
    @staticmethod
    @jit
    def std_to_nat(std_param):
        nat_param1 = jnp.einsum('nj, jm -> nm', std_param[1], std_param[0])
        nat_param2 = - 1/2 * std_param[1]
        return [nat_param1, nat_param2]

    @staticmethod
    @jit
    def nat_to_std(nat_param):
        mu = - 1/2 * jnp.einsum('nj, jm -> nm', inv(nat_param[1]), nat_param[0])
        lamda = - 2 * nat_param[1]
        return [mu, lamda]

    @staticmethod
    @jit
    def log_likelihood(std_param, x):
        ll_value = jnp.einsum('ml, mi, ik -> ', std_param[0], std_param[1], x) \
                   - 1/2 * jnp.einsum('mk, mi, ik -> ', x, std_param[1], x)

        ll_value -= 1/2 * jnp.einsum('mk, mi, ik -> ', std_param[0], std_param[1], std_param[0]) \
                    - jnp.sum(jnp.log(jnp.diag(jnp.linalg.cholesky(std_param[1]))))

        return ll_value - x.shape[1]/2 * jnp.log(2 * jnp.pi)


if __name__ == '__main__':
    @jit
    def loss(param, A, x):
        return -Gaussian.log_likelihood([param, A], x)


    A = jnp.eye(2)*100
    mu = jnp.array([1., 1.]).reshape(-1, 1)

    from jax.experimental import optimizers as opt
    from numpyro.distributions import MultivariateNormal

    mvnrml = MultivariateNormal(precision_matrix=A)
    x = mvnrml.sample(jax.random.PRNGKey(0), sample_shape=(10000,)).T

    print()

    import matplotlib.pyplot as plt

    plt.plot(x[0], x[1], 'x')

    opt_init, opt_update, get_params = opt.adam(1e-1)
    param = mu
    opt_state = opt_init(param)
    opt_state_init = opt_state

    value, grad = jax.value_and_grad(loss)(param, A, x)
    for i in range(100):
        opt_state = opt_update(0, grad, opt_state)
        param = get_params(opt_state)
        value, grad = jax.value_and_grad(loss)(param, A, x)
    print(param)

    mvnrml = MultivariateNormal(loc=param[:, 0], precision_matrix=A)
    x = mvnrml.sample(jax.random.PRNGKey(0), sample_shape=(1000,)).T
    plt.plot(x[0], x[1], 'xr')
    plt.show()