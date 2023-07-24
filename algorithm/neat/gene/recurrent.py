from dataclasses import dataclass

import jax
from jax import Array, numpy as jnp, vmap

from .normal import NormalGene, NormalGeneConfig
from core import State, Genome
from utils import Activation, Aggregation, unflatten_conns


@dataclass(frozen=True)
class RecurrentGeneConfig(NormalGeneConfig):
    activate_times: int = 10

    def __post_init__(self):
        super().__post_init__()
        assert self.activate_times > 0


class RecurrentGene(NormalGene):

    @staticmethod
    def forward_transform(state: State, genome: Genome):
        u_conns = unflatten_conns(genome.nodes, genome.conns)

        # remove un-enable connections and remove enable attr
        conn_enable = jnp.where(~jnp.isnan(u_conns[0]), True, False)
        u_conns = jnp.where(conn_enable, u_conns[1:, :], jnp.nan)

        return genome.nodes, u_conns

    @staticmethod
    def create_forward(state: State, config: RecurrentGeneConfig):
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

        batch_act, batch_agg = vmap(act), vmap(agg)

        def forward(inputs, transform) -> Array:
            nodes, cons = transform

            input_idx = state.input_idx
            output_idx = state.output_idx

            N = nodes.shape[0]
            vals = jnp.full((N,), 0.)

            weights = cons[0, :]

            def body_func(i, values):
                values = values.at[input_idx].set(inputs)
                nodes_ins = values * weights.T
                values = batch_agg(nodes[:, 4], nodes_ins)  # z = agg(ins)
                values = values * nodes[:, 2] + nodes[:, 1]  # z = z * response + bias
                values = batch_act(nodes[:, 3], values)  # z = act(z)
                return values

            vals = jax.lax.fori_loop(0, config.activate_times, body_func, vals)
            return vals[output_idx]

        return forward
