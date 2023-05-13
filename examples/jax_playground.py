import jax
import jax.numpy as jnp
from jax import jit, vmap
from time_utils import using_cprofile
from time import time
#
import numpy as np
@jit
def fx(x):
    return jnp.arange(x, x + 10)
#
#
# # @jit
# def fy(z):
#     z1, z2 = z, z + 1
#     vmap_fx = vmap(fx)
#     return vmap_fx(z1, z2)
#
# @jit
# def test_while(num, init_val):
#     def cond_fun(carry):
#         i, cumsum = carry
#         return i < num
#
#     def body_fun(carry):
#         i, cumsum = carry
#         cumsum += i
#         return i + 1, cumsum
#
#     return jax.lax.while_loop(cond_fun, body_fun, (0, init_val))




# @using_cprofile
def main():
    print(fx(1))

    # vmap_f = vmap(fx, in_axes=(None, 0))
    # vmap_vmap_f = vmap(vmap_f, in_axes=(0, None))
    # a = jnp.array([20,10,30])
    # b = jnp.array([6, 5, 4])
    # res = vmap_vmap_f(a, b)
    # print(res)
    # print(jnp.argmin(res, axis=1))



if __name__ == '__main__':
    main()
