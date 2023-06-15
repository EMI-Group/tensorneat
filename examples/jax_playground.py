import jax
import jax.numpy as jnp
from jax import jit, vmap
from time import time
import numpy as np


@jit
def jax_mutate(seed, x):
    noise = jax.random.normal(seed, x.shape) * 0.1
    return x + noise


def numpy_mutate(x):
    noise = np.random.normal(size=x.shape) * 0.1
    return x + noise


def jax_mutate_population(seed, pop_x):
    seeds = jax.random.split(seed, len(pop_x))
    func = vmap(jax_mutate, in_axes=(0, 0))
    return func(seeds, pop_x)


def numpy_mutate_population(pop_x):
    return np.stack([numpy_mutate(x) for x in pop_x])


def numpy_mutate_population_vmap(pop_x):
    noise = np.random.normal(size=pop_x.shape) * 0.1
    return pop_x + noise


def main():
    seed = jax.random.PRNGKey(0)
    i = 10
    while i < 200000:
        pop_x = jnp.ones((i, 100, 100))
        jax_pop_func = jit(jax_mutate_population).lower(seed, pop_x).compile()

        tic = time()
        res = jax.device_get(jax_pop_func(seed, pop_x))
        jax_time = time() - tic

        tic = time()
        res = numpy_mutate_population(pop_x)
        numpy_time = time() - tic

        tic = time()
        res = numpy_mutate_population_vmap(pop_x)
        numpy_time_vmap = time() - tic

        # print(f'POP_SIZE: {i} | JAX: {jax_time:.4f} | Numpy: {numpy_time:.4f} | Speedup: {numpy_time / jax_time:.4f}')
        print(f'POP_SIZE: {i} | JAX: {jax_time:.4f} | Numpy: {numpy_time:.4f} | Numpy Vmap: {numpy_time_vmap:.4f}')

        i = int(i * 1.3)


if __name__ == '__main__':
    main()
