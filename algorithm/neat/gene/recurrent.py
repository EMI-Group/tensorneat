from dataclasses import dataclass

import jax
from jax import numpy as jnp, vmap

from .normal import NormalGene, NormalGeneConfig
from core import State, Genome
from utils import unflatten_conns, act, agg


@dataclass(frozen=True)
class RecurrentGeneConfig(NormalGeneConfig):
    activate_times: int = 10

    def __post_init__(self):
        super().__post_init__()
        assert self.activate_times > 0


class RecurrentGene(NormalGene):

    def __init__(self, config: RecurrentGeneConfig = RecurrentGeneConfig()):
        self.config = config
        super().__init__(config)

    def forward_transform(self, state: State, genome: Genome):
        u_conns = unflatten_conns(genome.nodes, genome.conns)

        # remove un-enable connections and remove enable attr
        conn_enable = jnp.where(~jnp.isnan(u_conns[0]), True, False)
        u_conns = jnp.where(conn_enable, u_conns[1:, :], jnp.nan)

        return genome.nodes, u_conns

    def forward(self, state: State, inputs, transformed):
        nodes, conns = transformed

        batch_act, batch_agg = vmap(act, in_axes=(0, 0, None)), vmap(agg, in_axes=(0, 0, None))

        input_idx = state.input_idx
        output_idx = state.output_idx

        N = nodes.shape[0]
        vals = jnp.full((N,), 0.)

        weights = conns[0, :]

        def body_func(i, values):
            values = values.at[input_idx].set(inputs)
            nodes_ins = values * weights.T
            values = batch_agg(nodes[:, 4], nodes_ins, self.config.aggregation_options)  # z = agg(ins)
            values = values * nodes[:, 2] + nodes[:, 1]  # z = z * response + bias
            values = batch_act(nodes[:, 3], values, self.config.activation_options)  # z = act(z)
            return values

        vals = jax.lax.fori_loop(0, self.config.activate_times, body_func, vals)
        return vals[output_idx]
