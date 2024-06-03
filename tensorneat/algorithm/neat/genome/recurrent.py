from typing import Callable

import jax, jax.numpy as jnp
from utils import unflatten_conns

from . import BaseGenome
from ..gene import BaseNodeGene, BaseConnGene, DefaultNodeGene, DefaultConnGene
from ..ga import BaseMutation, BaseCrossover, DefaultMutation, DefaultCrossover


class RecurrentGenome(BaseGenome):
    """Default genome class, with the same behavior as the NEAT-Python"""

    network_type = "recurrent"

    def __init__(
        self,
        num_inputs: int,
        num_outputs: int,
        max_nodes: int,
        max_conns: int,
        node_gene: BaseNodeGene = DefaultNodeGene(),
        conn_gene: BaseConnGene = DefaultConnGene(),
        mutation: BaseMutation = DefaultMutation(),
        crossover: BaseCrossover = DefaultCrossover(),
        activate_time: int = 10,
        output_transform: Callable = None,
    ):
        super().__init__(
            num_inputs,
            num_outputs,
            max_nodes,
            max_conns,
            node_gene,
            conn_gene,
            mutation,
            crossover,
        )
        self.activate_time = activate_time

        if output_transform is not None:
            try:
                _ = output_transform(jnp.zeros(num_outputs))
            except Exception as e:
                raise ValueError(f"Output transform function failed: {e}")
        self.output_transform = output_transform

    def transform(self, state, nodes, conns):
        u_conns = unflatten_conns(nodes, conns)
        return nodes, conns, u_conns

    def restore(self, state, transformed):
        nodes, conns, u_conns = transformed
        return nodes, conns

    def forward(self, state, inputs, transformed):
        nodes, conns = transformed

        vals = jnp.full((self.max_nodes,), jnp.nan)
        nodes_attrs = nodes[:, 1:]  # remove index

        def body_func(_, values):

            # set input values
            values = values.at[self.input_idx].set(inputs)

            # calculate connections
            node_ins = jax.vmap(
                jax.vmap(self.conn_gene.forward, in_axes=(None, 1, None)),
                in_axes=(None, 1, 0),
            )(state, conns, values)

            # calculate nodes
            is_output_nodes = jnp.isin(jnp.arange(self.max_nodes), self.output_idx)
            values = jax.vmap(self.node_gene.forward, in_axes=(None, 0, 0, 0))(
                state, nodes_attrs, node_ins.T, is_output_nodes
            )

            return values

        vals = jax.lax.fori_loop(0, self.activate_time, body_func, vals)

        if self.output_transform is None:
            return vals[self.output_idx]
        else:
            return self.output_transform(vals[self.output_idx])
