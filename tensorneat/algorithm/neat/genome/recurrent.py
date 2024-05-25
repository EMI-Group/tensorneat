from typing import Callable

import jax, jax.numpy as jnp
from utils import unflatten_conns

from . import BaseGenome
from ..gene import BaseNodeGene, BaseConnGene, DefaultNodeGene, DefaultConnGene


class RecurrentGenome(BaseGenome):
    """Default genome class, with the same behavior as the NEAT-Python"""

    network_type = 'recurrent'

    def __init__(self,
                 num_inputs: int,
                 num_outputs: int,
                 max_nodes: int,
                 max_conns: int,
                 node_gene: BaseNodeGene = DefaultNodeGene(),
                 conn_gene: BaseConnGene = DefaultConnGene(),
                 activate_time: int = 10,
                 output_transform: Callable = None
                 ):
        super().__init__(num_inputs, num_outputs, max_nodes, max_conns, node_gene, conn_gene)
        self.activate_time = activate_time

        if output_transform is not None:
            try:
                _ = output_transform(jnp.zeros(num_outputs))
            except Exception as e:
                raise ValueError(f"Output transform function failed: {e}")
        self.output_transform = output_transform

    def transform(self, state, nodes, conns):
        u_conns = unflatten_conns(nodes, conns)

        # remove un-enable connections and remove enable attr
        conn_enable = u_conns[0] == 1
        u_conns = jnp.where(conn_enable, u_conns[1:, :], jnp.nan)

        return state, nodes, u_conns

    def forward(self, state, inputs, transformed):
        nodes, conns = transformed

        N = nodes.shape[0]
        vals = jnp.full((N,), jnp.nan)
        nodes_attrs = nodes[:, 1:]

        def body_func(_, carry):
            state_, values = carry

            # set input values
            values = values.at[self.input_idx].set(inputs)

            # calculate connections
            state_, node_ins = jax.vmap(
                jax.vmap(
                    self.conn_gene.forward,
                    in_axes=(None, 1, None),
                    out_axes=(None, 0)
                ),
                in_axes=(None, 1, 0),
                out_axes=(None, 0)
            )(state_, conns, values)

            # calculate nodes
            is_output_nodes = jnp.isin(
                jnp.arange(N),
                self.output_idx
            )
            state_, values = jax.vmap(
                self.node_gene.forward,
                in_axes=(None, 0, 0, 0),
                out_axes=(None, 0)
            )(state_, nodes_attrs, node_ins.T, is_output_nodes)

            return state_, values

        state, vals = jax.lax.fori_loop(0, self.activate_time, body_func, (state, vals))

        return state, vals[self.output_idx]
