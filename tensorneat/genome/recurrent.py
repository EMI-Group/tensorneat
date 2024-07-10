import jax
from jax import vmap, numpy as jnp
from .utils import unflatten_conns

from .base import BaseGenome
from .gene import DefaultNodeGene, DefaultConnGene
from .operations import DefaultMutation, DefaultCrossover, DefaultDistance
from .utils import unflatten_conns, extract_node_attrs, extract_conn_attrs

from tensorneat.common import attach_with_inf

class RecurrentGenome(BaseGenome):
    """Default genome class, with the same behavior as the NEAT-Python"""

    network_type = "recurrent"

    def __init__(
        self,
        num_inputs: int,
        num_outputs: int,
        max_nodes=50,
        max_conns=100,
        node_gene=DefaultNodeGene(),
        conn_gene=DefaultConnGene(),
        mutation=DefaultMutation(),
        crossover=DefaultCrossover(),
        distance=DefaultDistance(),
        output_transform=None,
        input_transform=None,
        init_hidden_layers=(),
        activate_time=10,
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
            distance,
            output_transform,
            input_transform,
            init_hidden_layers,
        )
        self.activate_time = activate_time

    def transform(self, state, nodes, conns):
        u_conns = unflatten_conns(nodes, conns)
        return nodes, conns, u_conns

    def forward(self, state, transformed, inputs):
        nodes, conns, u_conns = transformed

        vals = jnp.full((self.max_nodes,), jnp.nan)

        nodes_attrs = vmap(extract_node_attrs)(nodes)
        conns_attrs = vmap(extract_conn_attrs)(conns)
        expand_conns_attrs = attach_with_inf(conns_attrs, u_conns)

        def body_func(_, values):

            # set input values
            values = values.at[self.input_idx].set(inputs)

            # calculate connections
            node_ins = vmap(
                vmap(self.conn_gene.forward, in_axes=(None, 0, None)),
                in_axes=(None, 0, 0),
            )(state, expand_conns_attrs, values)

            # calculate nodes
            is_output_nodes = jnp.isin(nodes[:, 0], self.output_idx)
            values = vmap(self.node_gene.forward, in_axes=(None, 0, 0, 0))(
                state, nodes_attrs, node_ins.T, is_output_nodes
            )

            return values

        vals = jax.lax.fori_loop(0, self.activate_time, body_func, vals)

        if self.output_transform is None:
            return vals[self.output_idx]
        else:
            return self.output_transform(vals[self.output_idx])

    def sympy_func(self, state, network, precision=3):
        raise ValueError("Sympy function is not supported for Recurrent Network!")

    def visualize(self, network):
        raise ValueError("Visualize function is not supported for Recurrent Network!")
