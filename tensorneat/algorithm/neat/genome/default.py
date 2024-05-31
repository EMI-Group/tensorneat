from typing import Callable

import jax, jax.numpy as jnp
from utils import unflatten_conns, flatten_conns, topological_sort, I_INF

from . import BaseGenome
from ..gene import BaseNodeGene, BaseConnGene, DefaultNodeGene, DefaultConnGene
from ..ga import BaseMutation, BaseCrossover, DefaultMutation, DefaultCrossover


class DefaultGenome(BaseGenome):
    """Default genome class, with the same behavior as the NEAT-Python"""

    network_type = "feedforward"

    def __init__(
        self,
        num_inputs: int,
        num_outputs: int,
        max_nodes=5,
        max_conns=4,
        node_gene: BaseNodeGene = DefaultNodeGene(),
        conn_gene: BaseConnGene = DefaultConnGene(),
        mutation: BaseMutation = DefaultMutation(),
        crossover: BaseCrossover = DefaultCrossover(),
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

        if output_transform is not None:
            try:
                _ = output_transform(jnp.zeros(num_outputs))
            except Exception as e:
                raise ValueError(f"Output transform function failed: {e}")
        self.output_transform = output_transform

    def transform(self, state, nodes, conns):
        u_conns = unflatten_conns(nodes, conns)
        conn_exist = ~jnp.isnan(u_conns[0])

        seqs = topological_sort(nodes, conn_exist)

        return seqs, nodes, u_conns

    def restore(self, state, transformed):
        seqs, nodes, u_conns = transformed
        conns = flatten_conns(nodes, u_conns, C=self.max_conns)
        return nodes, conns

    def forward(self, state, inputs, transformed):
        cal_seqs, nodes, u_conns = transformed

        ini_vals = jnp.full((self.max_nodes,), jnp.nan)
        ini_vals = ini_vals.at[self.input_idx].set(inputs)
        nodes_attrs = nodes[:, 1:]

        def cond_fun(carry):
            values, idx = carry
            return (idx < self.max_nodes) & (cal_seqs[idx] != I_INF)

        def body_func(carry):
            values, idx = carry
            i = cal_seqs[idx]

            def hit():
                ins = jax.vmap(self.conn_gene.forward, in_axes=(None, 1, 0))(
                    state, u_conns[:, :, i], values
                )

                z = self.node_gene.forward(
                    state,
                    nodes_attrs[i],
                    ins,
                    is_output_node=jnp.isin(i, self.output_idx),
                )

                new_values = values.at[i].set(z)
                return new_values

            # the val of input nodes is obtained by the task, not by calculation
            values = jax.lax.cond(jnp.isin(i, self.input_idx), lambda: values, hit)

            return values, idx + 1

        vals, _ = jax.lax.while_loop(cond_fun, body_func, (ini_vals, 0))

        if self.output_transform is None:
            return vals[self.output_idx]
        else:
            return self.output_transform(vals[self.output_idx])

    def update_by_batch(self, state, batch_input, transformed):
        cal_seqs, nodes, u_conns = transformed

        batch_size = batch_input.shape[0]
        batch_ini_vals = jnp.full((batch_size, self.max_nodes), jnp.nan)
        batch_ini_vals = batch_ini_vals.at[:, self.input_idx].set(batch_input)
        nodes_attrs = nodes[:, 1:]

        def cond_fun(carry):
            batch_values, nodes_attrs_, u_conns_, idx = carry
            return (idx < self.max_nodes) & (cal_seqs[idx] != I_INF)

        def body_func(carry):
            batch_values, nodes_attrs_, u_conns_, idx = carry
            i = cal_seqs[idx]

            def hit():
                batch_ins, new_conn_attrs = jax.vmap(
                    self.conn_gene.update_by_batch,
                    in_axes=(None, 1, 1),
                    out_axes=(1, 1),
                )(state, u_conns_[:, :, i], batch_values)
                batch_z, new_node_attrs = self.node_gene.update_by_batch(
                    state,
                    nodes_attrs[i],
                    batch_ins,
                    is_output_node=jnp.isin(i, self.output_idx),
                )
                new_batch_values = batch_values.at[:, i].set(batch_z)
                return (
                    new_batch_values,
                    nodes_attrs_.at[i].set(new_node_attrs),
                    u_conns_.at[:, :, i].set(new_conn_attrs),
                )

            # the val of input nodes is obtained by the task, not by calculation
            (batch_values, nodes_attrs_, u_conns_) = jax.lax.cond(
                jnp.isin(i, self.input_idx),
                lambda: (batch_values, nodes_attrs_, u_conns_),
                hit,
            )

            return batch_values, nodes_attrs_, u_conns_, idx + 1

        batch_vals, nodes_attrs, u_conns, _ = jax.lax.while_loop(
            cond_fun, body_func, (batch_ini_vals, nodes_attrs, u_conns, 0)
        )

        nodes = nodes.at[:, 1:].set(nodes_attrs)
        new_transformed = (cal_seqs, nodes, u_conns)

        if self.output_transform is None:
            return batch_vals[:, self.output_idx], new_transformed
        else:
            return (
                jax.vmap(self.output_transform)(batch_vals[:, self.output_idx]),
                new_transformed,
            )
