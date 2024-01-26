import jax, jax.numpy as jnp
from utils import unflatten_conns, topological_sort, I_INT

from . import BaseGenome
from ..gene import BaseNodeGene, BaseConnGene, DefaultNodeGene, DefaultConnGene


class DefaultGenome(BaseGenome):
    """Default genome class, with the same behavior as the NEAT-Python"""

    network_type = 'feedforward'

    def __init__(self,
                 num_inputs: int,
                 num_outputs: int,
                 node_gene: BaseNodeGene = DefaultNodeGene(),
                 conn_gene: BaseConnGene = DefaultConnGene(),
                 ):
        super().__init__(num_inputs, num_outputs, node_gene, conn_gene)

    def transform(self, nodes, conns):
        u_conns = unflatten_conns(nodes, conns)

        # DONE: Seems like there is a bug in this line
        # conn_enable = jnp.where(~jnp.isnan(u_conns[0]), True, False)
        # modified: exist conn and enable is true
        # conn_enable = jnp.where( (~jnp.isnan(u_conns[0])) & (u_conns[0] == 1), True, False)
        # advanced modified: when and only when enabled is True
        conn_enable = u_conns[0] == 1

        # remove enable attr
        u_conns = jnp.where(conn_enable, u_conns[1:, :], jnp.nan)
        seqs = topological_sort(nodes, conn_enable)

        return seqs, nodes, u_conns

    def forward(self, inputs, transformed):
        cal_seqs, nodes, conns = transformed

        N = nodes.shape[0]
        ini_vals = jnp.full((N,), jnp.nan)
        ini_vals = ini_vals.at[self.input_idx].set(inputs)
        nodes_attrs = nodes[:, 1:]

        def cond_fun(carry):
            values, idx = carry
            return (idx < N) & (cal_seqs[idx] != I_INT)

        def body_func(carry):
            values, idx = carry
            i = cal_seqs[idx]

            def hit():
                ins = jax.vmap(self.conn_gene.forward, in_axes=(1, 0))(conns[:, :, i], values)
                # ins = values * weights[:, i]

                z = self.node_gene.forward(nodes_attrs[i], ins)
                # z = agg(nodes[i, 4], ins, self.config.aggregation_options)  # z = agg(ins)
                # z = z * nodes[i, 2] + nodes[i, 1]  # z = z * response + bias
                # z = act(nodes[i, 3], z, self.config.activation_options)  # z = act(z)

                new_values = values.at[i].set(z)
                return new_values

            def miss():
                return values

            # the val of input nodes is obtained by the task, not by calculation
            values = jax.lax.cond(jnp.isin(i, self.input_idx), miss, hit)

            return values, idx + 1

        vals, _ = jax.lax.while_loop(cond_fun, body_func, (ini_vals, 0))

        return vals[self.output_idx]
