import jax
from jax import Array, numpy as jnp

from . import BaseGene


class NormalGene(BaseGene):
    node_attrs = ['bias', 'response', 'aggregation', 'activation']
    conn_attrs = ['weight']

    @staticmethod
    def setup(state, config):
        return state.update(
            bias_init_mean=config['bias_init_mean'],
            bias_init_std=config['bias_init_std'],
            bias_mutate_power=config['bias_mutate_power'],
            bias_mutate_rate=config['bias_mutate_rate'],
            bias_replace_rate=config['bias_replace_rate'],

            response_init_mean=config['response_init_mean'],
            response_init_std=config['response_init_std'],
            response_mutate_power=config['response_mutate_power'],
            response_mutate_rate=config['response_mutate_rate'],
            response_replace_rate=config['response_replace_rate'],

            activation_default=config['activation_default'],
            activation_options=config['activation_options'],
            activation_replace_rate=config['activation_replace_rate'],

            aggregation_default=config['aggregation_default'],
            aggregation_options=config['aggregation_options'],
            aggregation_replace_rate=config['aggregation_replace_rate'],

            weight_init_mean=config['weight_init_mean'],
            weight_init_std=config['weight_init_std'],
            weight_mutate_power=config['weight_mutate_power'],
            weight_mutate_rate=config['weight_mutate_rate'],
            weight_replace_rate=config['weight_replace_rate'],
        )

    @staticmethod
    def new_node_attrs(state):
        return jnp.array([state.bias_init_mean, state.response_init_mean,
                          state.activation_default, state.aggregation_default])

    @staticmethod
    def new_conn_attrs(state):
        return jnp.array([state.weight_init_mean])

    @staticmethod
    def mutate_node(state, attrs: Array, key):
        k1, k2, k3, k4 = jax.random.split(key, num=4)

        bias = NormalGene._mutate_float(k1, attrs[0], state.bias_init_mean, state.bias_init_std,
                                        state.bias_mutate_power, state.bias_mutate_rate, state.bias_replace_rate)
        res = NormalGene._mutate_float(k2, attrs[1], state.response_init_mean, state.response_init_std,
                                       state.response_mutate_power, state.response_mutate_rate,
                                       state.response_replace_rate)
        act = NormalGene._mutate_int(k3, attrs[2], state.activation_options, state.activation_replace_rate)
        agg = NormalGene._mutate_int(k4, attrs[3], state.aggregation_options, state.aggregation_replace_rate)

        return jnp.array([bias, res, act, agg])

    @staticmethod
    def mutate_conn(state, attrs: Array, key):
        weight = NormalGene._mutate_float(key, attrs[0], state.weight_init_mean, state.weight_init_std,
                                          state.weight_mutate_power, state.weight_mutate_rate,
                                          state.weight_replace_rate)

        return jnp.array([weight])

    @staticmethod
    def distance_node(state, array1: Array, array2: Array):
        # bias + response + activation + aggregation
        return jnp.abs(array1[1] - array2[1]) + jnp.abs(array1[2] - array2[2]) + \
            (array1[3] != array2[3]) + (array1[4] != array2[4])

    @staticmethod
    def distance_conn(state, array1: Array, array2: Array):
        return (array1[2] != array2[2]) + jnp.abs(array1[3] - array2[3])  # enable + weight

    @staticmethod
    def forward(state, array: Array):
        return array

    @staticmethod
    def _mutate_float(key, val, init_mean, init_std, mutate_power, mutate_rate, replace_rate):
        k1, k2, k3 = jax.random.split(key, num=3)
        noise = jax.random.normal(k1, ()) * mutate_power
        replace = jax.random.normal(k2, ()) * init_std + init_mean
        r = jax.random.uniform(k3, ())

        val = jnp.where(
            r < mutate_rate,
            val + noise,
            jnp.where(
                (mutate_rate < r) & (r < mutate_rate + replace_rate),
                replace,
                val
            )
        )

        return val

    @staticmethod
    def _mutate_int(key, val, options, replace_rate):
        k1, k2 = jax.random.split(key, num=2)
        r = jax.random.uniform(k1, ())

        val = jnp.where(
            r < replace_rate,
            jax.random.choice(k2, options),
            val
        )

        return val
