from typing import Callable

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
                 max_nodes=5,
                 max_conns=4,
                 node_gene: BaseNodeGene = DefaultNodeGene(),
                 conn_gene: BaseConnGene = DefaultConnGene(),
                 output_transform: Callable = None
                 ):
        super().__init__(num_inputs, num_outputs, max_nodes, max_conns, node_gene, conn_gene)

        if output_transform is not None:
            try:
                _ = output_transform(jnp.zeros(num_outputs))
            except Exception as e:
                raise ValueError(f"Output transform function failed: {e}")
        self.output_transform = output_transform

    def transform(self, state, nodes, conns):
        u_conns = unflatten_conns(nodes, conns)
        conn_enable = u_conns[0] == 1

        # remove enable attr
        u_conns = jnp.where(conn_enable, u_conns[1:, :], jnp.nan)
        seqs = topological_sort(nodes, conn_enable)

        return state, seqs, nodes, u_conns

    def forward(self, state, inputs, transformed):
        cal_seqs, nodes, conns = transformed

        N = nodes.shape[0]
        ini_vals = jnp.full((N,), jnp.nan)
        ini_vals = ini_vals.at[self.input_idx].set(inputs)
        nodes_attrs = nodes[:, 1:]

        def cond_fun(carry):
            state_, values, idx = carry
            return (idx < N) & (cal_seqs[idx] != I_INT)

        def body_func(carry):
            state_, values, idx = carry
            i = cal_seqs[idx]

            def hit():
                s, ins = jax.vmap(self.conn_gene.forward,
                                  in_axes=(None, 1, 0), out_axes=(None, 0))(state_, conns[:, :, i], values)
                s, z = self.node_gene.forward(s, nodes_attrs[i], ins, is_output_node=jnp.isin(i, self.output_idx))
                new_values = values.at[i].set(z)
                return s, new_values

            # the val of input nodes is obtained by the task, not by calculation
            state_, values = jax.lax.cond(
                jnp.isin(i, self.input_idx),
                lambda: (state_, values),
                hit
            )

            return state_, values, idx + 1

        state, vals, _ = jax.lax.while_loop(cond_fun, body_func, (state, ini_vals, 0))

        if self.output_transform is None:
            return state, vals[self.output_idx]
        else:
            return state, self.output_transform(vals[self.output_idx])
