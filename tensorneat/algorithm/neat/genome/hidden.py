import jax, jax.numpy as jnp
from .default import DefaultGenome


class HiddenInitialize(DefaultGenome):
    def __init__(self, hidden_cnt=8, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.hidden_cnt = hidden_cnt

    def initialize(self, state, randkey):

        k1, k2 = jax.random.split(randkey, num=2)

        input_idx, output_idx = self.input_idx, self.output_idx
        input_size = len(input_idx)
        output_size = len(output_idx)

        hidden_idx = jnp.arange(
            input_size + output_size, input_size + output_size + self.hidden_cnt
        )
        nodes = jnp.full(
            (self.max_nodes, self.node_gene.length), jnp.nan, dtype=jnp.float32
        )

        nodes = nodes.at[input_idx, 0].set(input_idx)
        nodes = nodes.at[output_idx, 0].set(output_idx)
        nodes = nodes.at[hidden_idx, 0].set(hidden_idx)

        total_idx = input_size + output_size + self.hidden_cnt
        rand_keys_n = jax.random.split(k1, num=total_idx)

        node_attr_func = jax.vmap(self.node_gene.new_random_attrs, in_axes=(None, 0))
        node_attrs = node_attr_func(state, rand_keys_n)
        nodes = nodes.at[:total_idx, 1:].set(node_attrs)

        conns = jnp.full(
            (self.max_conns, self.conn_gene.length), jnp.nan, dtype=jnp.float32
        )

        input_to_hidden_ids, hidden_ids = jnp.meshgrid(
            input_idx, hidden_idx, indexing="ij"
        )
        total_input_to_hidden_conns = input_size * self.hidden_cnt
        conns = conns.at[:total_input_to_hidden_conns, :2].set(
            jnp.column_stack([input_to_hidden_ids.flatten(), hidden_ids.flatten()])
        )

        hidden_to_output_ids, output_ids = jnp.meshgrid(
            hidden_idx, output_idx, indexing="ij"
        )
        total_hidden_to_output_conns = self.hidden_cnt * output_size
        conns = conns.at[
            total_input_to_hidden_conns : total_input_to_hidden_conns
            + total_hidden_to_output_conns,
            :2,
        ].set(jnp.column_stack([hidden_to_output_ids.flatten(), output_ids.flatten()]))

        total_conns = total_input_to_hidden_conns + total_hidden_to_output_conns
        rand_keys_c = jax.random.split(k2, num=total_conns)
        conns_attr_func = jax.vmap(
            self.conn_gene.new_random_attrs,
            in_axes=(
                None,
                0,
            ),
        )
        conns_attrs = conns_attr_func(state, rand_keys_c)
        conns = conns.at[:total_conns, 2:].set(conns_attrs)

        return nodes, conns
