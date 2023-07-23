from dataclasses import dataclass
from typing import Tuple

import jax
from jax import Array, numpy as jnp

from config import GeneConfig
from core import Gene, Genome, State
from utils import Activation, Aggregation, unflatten_conns, topological_sort, I_INT


@dataclass(frozen=True)
class NormalGeneConfig(GeneConfig):
    bias_init_mean: float = 0.0
    bias_init_std: float = 1.0
    bias_mutate_power: float = 0.5
    bias_mutate_rate: float = 0.7
    bias_replace_rate: float = 0.1

    response_init_mean: float = 1.0
    response_init_std: float = 0.0
    response_mutate_power: float = 0.5
    response_mutate_rate: float = 0.7
    response_replace_rate: float = 0.1

    activation_default: str = 'sigmoid'
    activation_options: Tuple[str] = ('sigmoid',)
    activation_replace_rate: float = 0.1

    aggregation_default: str = 'sum'
    aggregation_options: Tuple[str] = ('sum',)
    aggregation_replace_rate: float = 0.1

    weight_init_mean: float = 0.0
    weight_init_std: float = 1.0
    weight_mutate_power: float = 0.5
    weight_mutate_rate: float = 0.8
    weight_replace_rate: float = 0.1

    def __post_init__(self):
        assert self.bias_init_std >= 0.0
        assert self.bias_mutate_power >= 0.0
        assert self.bias_mutate_rate >= 0.0
        assert self.bias_replace_rate >= 0.0

        assert self.response_init_std >= 0.0
        assert self.response_mutate_power >= 0.0
        assert self.response_mutate_rate >= 0.0
        assert self.response_replace_rate >= 0.0

        assert self.activation_default == self.activation_options[0]

        for name in self.activation_options:
            assert name in Activation.name2func, f"Activation function: {name} not found"

        assert self.aggregation_default == self.aggregation_options[0]

        assert self.aggregation_default in Aggregation.name2func, \
            f"Aggregation function: {self.aggregation_default} not found"

        for name in self.aggregation_options:
            assert name in Aggregation.name2func, f"Aggregation function: {name} not found"


class NormalGene(Gene):
    node_attrs = ['bias', 'response', 'aggregation', 'activation']
    conn_attrs = ['weight']

    @staticmethod
    def setup(config: NormalGeneConfig, state: State = State()):

        return state.update(
            bias_init_mean=config.bias_init_mean,
            bias_init_std=config.bias_init_std,
            bias_mutate_power=config.bias_mutate_power,
            bias_mutate_rate=config.bias_mutate_rate,
            bias_replace_rate=config.bias_replace_rate,

            response_init_mean=config.response_init_mean,
            response_init_std=config.response_init_std,
            response_mutate_power=config.response_mutate_power,
            response_mutate_rate=config.response_mutate_rate,
            response_replace_rate=config.response_replace_rate,

            activation_replace_rate=config.activation_replace_rate,
            activation_default=0,
            activation_options=jnp.arange(len(config.activation_options)),

            aggregation_replace_rate=config.aggregation_replace_rate,
            aggregation_default=0,
            aggregation_options=jnp.arange(len(config.aggregation_options)),

            weight_init_mean=config.weight_init_mean,
            weight_init_std=config.weight_init_std,
            weight_mutate_power=config.weight_mutate_power,
            weight_mutate_rate=config.weight_mutate_rate,
            weight_replace_rate=config.weight_replace_rate,
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
    def distance_node(state, node1: Array, node2: Array):
        # bias + response + activation + aggregation
        return jnp.abs(node1[1] - node2[1]) + jnp.abs(node1[2] - node2[2]) + \
            (node1[3] != node2[3]) + (node1[4] != node2[4])

    @staticmethod
    def distance_conn(state, con1: Array, con2: Array):
        return (con1[2] != con2[2]) + jnp.abs(con1[3] - con2[3])  # enable + weight

    @staticmethod
    def forward_transform(state: State, genome: Genome):
        u_conns = unflatten_conns(genome.nodes, genome.conns)
        conn_enable = jnp.where(~jnp.isnan(u_conns[0]), True, False)

        # remove enable attr
        u_conns = jnp.where(conn_enable, u_conns[1:, :], jnp.nan)
        seqs = topological_sort(genome.nodes, conn_enable)

        return seqs, genome.nodes, u_conns

    @staticmethod
    def create_forward(state: State, config: NormalGeneConfig):
        activation_funcs = [Activation.name2func[name] for name in config.activation_options]
        aggregation_funcs = [Aggregation.name2func[name] for name in config.aggregation_options]

        def act(idx, z):
            """
            calculate activation function for each node
            """
            idx = jnp.asarray(idx, dtype=jnp.int32)
            # change idx from float to int
            res = jax.lax.switch(idx, activation_funcs, z)
            return res

        def agg(idx, z):
            """
            calculate activation function for inputs of node
            """
            idx = jnp.asarray(idx, dtype=jnp.int32)

            def all_nan():
                return 0.

            def not_all_nan():
                return jax.lax.switch(idx, aggregation_funcs, z)

            return jax.lax.cond(jnp.all(jnp.isnan(z)), all_nan, not_all_nan)

        def forward(inputs, transformed) -> Array:
            """
            forward for single input shaped (input_num, )

            :argument inputs: (input_num, )
            :argument cal_seqs: (N, )
            :argument nodes: (N, 5)
            :argument connections: (2, N, N)

            :return (output_num, )
            """

            cal_seqs, nodes, cons = transformed

            input_idx = state.input_idx
            output_idx = state.output_idx

            N = nodes.shape[0]
            ini_vals = jnp.full((N,), jnp.nan)
            ini_vals = ini_vals.at[input_idx].set(inputs)

            weights = cons[0, :]

            def cond_fun(carry):
                values, idx = carry
                return (idx < N) & (cal_seqs[idx] != I_INT)

            def body_func(carry):
                values, idx = carry
                i = cal_seqs[idx]

                def hit():
                    ins = values * weights[:, i]
                    z = agg(nodes[i, 4], ins)  # z = agg(ins)
                    z = z * nodes[i, 2] + nodes[i, 1]  # z = z * response + bias
                    z = act(nodes[i, 3], z)  # z = act(z)

                    new_values = values.at[i].set(z)
                    return new_values

                def miss():
                    return values

                # the val of input nodes is obtained by the task, not by calculation
                values = jax.lax.cond(jnp.isin(i, input_idx), miss, hit)

                return values, idx + 1

            vals, _ = jax.lax.while_loop(cond_fun, body_func, (ini_vals, 0))

            return vals[output_idx]

        return forward

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
