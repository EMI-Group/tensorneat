import jax
import jax.numpy as jnp


class Agg:
    @staticmethod
    def name2func(name):
        return getattr(Agg, name)

    @staticmethod
    def sum(z):
        return jnp.sum(z, axis=0, where=~jnp.isnan(z))

    @staticmethod
    def product(z):
        return jnp.prod(z, axis=0, where=~jnp.isnan(z))

    @staticmethod
    def max(z):
        return jnp.max(z, axis=0, where=~jnp.isnan(z))

    @staticmethod
    def min(z):
        return jnp.min(z, axis=0, where=~jnp.isnan(z))

    @staticmethod
    def maxabs(z):
        z = jnp.where(jnp.isnan(z), 0, z)
        abs_z = jnp.abs(z)
        max_abs_index = jnp.argmax(abs_z)
        return z[max_abs_index]

    @staticmethod
    def median(z):
        n = jnp.sum(~jnp.isnan(z), axis=0)

        z = jnp.sort(z)  # sort

        idx1, idx2 = (n - 1) // 2, n // 2
        median = (z[idx1] + z[idx2]) / 2

        return median

    @staticmethod
    def mean(z):
        aux = jnp.where(jnp.isnan(z), 0, z)
        valid_values_sum = jnp.sum(aux, axis=0)
        valid_values_count = jnp.sum(~jnp.isnan(z), axis=0)
        mean_without_zeros = valid_values_sum / valid_values_count
        return mean_without_zeros


AGG_ALL = (Agg.sum, Agg.product, Agg.max, Agg.min, Agg.maxabs, Agg.median, Agg.mean)


def agg_func(idx, z, agg_funcs):
    """
    calculate activation function for inputs of node
    """
    idx = jnp.asarray(idx, dtype=jnp.int32)

    return jax.lax.cond(
        jnp.all(jnp.isnan(z)),
        lambda: jnp.nan,  # all inputs are nan
        lambda: jax.lax.switch(idx, agg_funcs, z),  # otherwise
    )
