import jax.numpy as jnp


class Aggregation:

    @staticmethod
    def sum_agg(z):
        z = jnp.where(jnp.isnan(z), 0, z)
        return jnp.sum(z, axis=0)

    @staticmethod
    def product_agg(z):
        z = jnp.where(jnp.isnan(z), 1, z)
        return jnp.prod(z, axis=0)

    @staticmethod
    def max_agg(z):
        z = jnp.where(jnp.isnan(z), -jnp.inf, z)
        return jnp.max(z, axis=0)

    @staticmethod
    def min_agg(z):
        z = jnp.where(jnp.isnan(z), jnp.inf, z)
        return jnp.min(z, axis=0)

    @staticmethod
    def maxabs_agg(z):
        z = jnp.where(jnp.isnan(z), 0, z)
        abs_z = jnp.abs(z)
        max_abs_index = jnp.argmax(abs_z)
        return z[max_abs_index]

    @staticmethod
    def median_agg(z):
        n = jnp.sum(~jnp.isnan(z), axis=0)

        z = jnp.sort(z)  # sort

        idx1, idx2 = (n - 1) // 2, n // 2
        median = (z[idx1] + z[idx2]) / 2

        return median

    @staticmethod
    def mean_agg(z):
        aux = jnp.where(jnp.isnan(z), 0, z)
        valid_values_sum = jnp.sum(aux, axis=0)
        valid_values_count = jnp.sum(~jnp.isnan(z), axis=0)
        mean_without_zeros = valid_values_sum / valid_values_count
        return mean_without_zeros

    name2func = {
        'sum': sum_agg,
        'product': product_agg,
        'max': max_agg,
        'min': min_agg,
        'maxabs': maxabs_agg,
        'median': median_agg,
        'mean': mean_agg,
    }
