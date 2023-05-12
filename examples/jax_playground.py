import jax
import jax.numpy as jnp
from jax import jit, vmap
from time_utils import using_cprofile
from time import time

@jit
def fx(x, y):
    return x + y


# @jit
def fy(z):
    z1, z2 = z, z + 1
    vmap_fx = vmap(fx)
    return vmap_fx(z1, z2)

@jit
def test_while(num, init_val):
    def cond_fun(carry):
        i, cumsum = carry
        return i < num

    def body_fun(carry):
        i, cumsum = carry
        cumsum += i
        return i + 1, cumsum

    return jax.lax.while_loop(cond_fun, body_fun, (0, init_val))



@using_cprofile
def main():
    z = jnp.zeros((100000, ))

    num = 100

    nums = jnp.arange(num) * 10

    f = jit(vmap(test_while, in_axes=(0, None))).lower(nums, z).compile()
    def test_time(*args):
        return f(*args)

    print(test_time(nums, z))

    #
    #
    # for i in range(10):
    #     num = 10 ** i
    #     st = time()
    #     res = test_time(num, z)
    #     print(res)
    #     t = time() - st
    #     print(f"num: {num}, time: {t}")

if __name__ == '__main__':
    main()
