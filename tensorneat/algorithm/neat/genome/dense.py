import jax, jax.numpy as jnp
from .default import DefaultGenome


class DenseInitialize(DefaultGenome):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert self.max_nodes >= self.num_inputs + self.num_outputs
        assert self.max_conns >= self.num_inputs * self.num_outputs

    def initialize(self, state, randkey):

        k1, k2 = jax.random.split(randkey, num=2)

        input_idx, output_idx = self.input_idx, self.output_idx
        input_size = len(input_idx)
        output_size = len(output_idx)

        nodes = jnp.full(
            (self.max_nodes, self.node_gene.length), jnp.nan, dtype=jnp.float32
        )

        nodes = nodes.at[input_idx, 0].set(input_idx)
        nodes = nodes.at[output_idx, 0].set(output_idx)

        total_idx = input_size + output_size
        rand_keys_n = jax.random.split(k1, num=total_idx)

        node_attr_func = jax.vmap(self.node_gene.new_random_attrs, in_axes=(None, 0))
        node_attrs = node_attr_func(state, rand_keys_n)
        nodes = nodes.at[:total_idx, 1:].set(node_attrs)

        conns = jnp.full(
            (self.max_conns, self.conn_gene.length), jnp.nan, dtype=jnp.float32
        )

        input_to_output_ids, output_ids = jnp.meshgrid(
            input_idx, output_idx, indexing="ij"
        )
        total_conns = input_size * output_size
        conns = conns.at[:total_conns, :2].set(
            jnp.column_stack([input_to_output_ids.flatten(), output_ids.flatten()])
        )

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
