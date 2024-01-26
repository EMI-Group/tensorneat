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
                 node_gene: BaseNodeGene = DefaultNodeGene(),
                 conn_gene: BaseConnGene = DefaultConnGene(),
                 activate_time: int = 10,
                 ):
        super().__init__(num_inputs, num_outputs, node_gene, conn_gene)
        self.activate_time = activate_time

    def transform(self, nodes, conns):
        u_conns = unflatten_conns(nodes, conns)

        # remove un-enable connections and remove enable attr
        conn_enable = u_conns[0] == 1
        u_conns = jnp.where(conn_enable, u_conns[1:, :], jnp.nan)

        return nodes, u_conns

    def forward(self, inputs, transformed):
        nodes, conns = transformed

        N = nodes.shape[0]
        vals = jnp.full((N,), jnp.nan)
        nodes_attrs = nodes[:, 1:]

        def body_func(_, values):
            # set input values
            values = values.at[self.input_idx].set(inputs)

            # calculate connections
            node_ins = jax.vmap(
                jax.vmap(
                    self.conn_gene.forward,
                    in_axes=(1, None)
                ),
                in_axes=(1, 0)
            )(conns, values)

            # calculate nodes
            values = jax.vmap(self.node_gene.forward)(nodes_attrs, node_ins.T)
            return values

        vals = jax.lax.fori_loop(0, self.activate_time, body_func, vals)

        return vals[self.output_idx]
