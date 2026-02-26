import jax
from jax import vmap, numpy as jnp
from mujoco import rollout

from .default import DefaultGenome
from .gene import RNNNode, DefaultConn
from .operations import DefaultMutation, DefaultCrossover, DefaultDistance
from .utils import extract_gene_attrs

from tensorneat.common import I_INF, attach_with_inf

class StatefulDefaultGenome(DefaultGenome):
    def __init__(
        self,
        num_inputs: int,
        num_outputs: int,
        max_nodes=50,
        max_conns=100,
        node_gene=RNNNode(),
        conn_gene=DefaultConn(),
        mutation=DefaultMutation(),
        crossover=DefaultCrossover(),
        distance=DefaultDistance(),
        output_transform=None,
        input_transform=None,
        init_hidden_layers=(),
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

    def forward(self, state, transformed, inputs, rollout_state):

        if self.input_transform is not None:
            inputs = self.input_transform(inputs)

        cal_seqs, nodes, conns, u_conns = transformed

        ini_vals = jnp.full((self.max_nodes,), jnp.nan)
        ini_vals = ini_vals.at[self.input_idx].set(inputs)
        nodes_attrs = vmap(extract_gene_attrs, in_axes=(None, 0))(self.node_gene, nodes)
        conns_attrs = vmap(extract_gene_attrs, in_axes=(None, 0))(self.conn_gene, conns)

        def cond_fun(carry):
            values, rollout_state, idx = carry
            return (idx < self.max_nodes) & (
                cal_seqs[idx] != I_INF
            )  # not out of bounds and next node exists

        def body_func(carry):
            values, rollout_state, idx = carry
            i = cal_seqs[idx]

            def input_node():
                return values, rollout_state

            def otherwise():
                # calculate connections
                conn_indices = u_conns[:, i]
                hit_attrs = attach_with_inf(
                    conns_attrs, conn_indices
                )  # fetch conn attrs
                ins = vmap(self.conn_gene.forward, in_axes=(None, 0, 0))(
                    state, hit_attrs, values
                )
                # Add rollout/hidden state
                ins = jnp.append(rollout_state[i], ins)

                # calculate nodes
                z = self.node_gene.forward(
                    state,
                    nodes_attrs[i],
                    ins,
                    is_output_node=jnp.isin(
                        nodes[i, 0], self.output_idx
                    ),  # nodes[0] -> the key of nodes
                )
                output, new_node_rollout_state = z

                # set new value
                new_values = values.at[i].set(output)
                new_rollout_state = rollout_state.at[i].set(new_node_rollout_state)
                return new_values, new_rollout_state

            values, rollout_state = jax.lax.cond(jnp.isin(i, self.input_idx), input_node, otherwise)

            return values, rollout_state, idx + 1

        vals, rollout_state, _ = jax.lax.while_loop(cond_fun, body_func, (ini_vals, rollout_state, 0))

        if self.output_transform is None:
            return vals[self.output_idx], rollout_state
        else:
            return self.output_transform(vals[self.output_idx]), rollout_state
        
    def init_rollout_state(self, state, params):
        return jnp.zeros((self.max_nodes,))