import jax
from jax import numpy as jnp, vmap

from algorithm.neat import BaseGene
from algorithm.neat.gene import Activation
from algorithm.neat.gene import Aggregation


class HyperNEATGene(BaseGene):
    node_attrs = []  # no node attributes
    conn_attrs = ['weight']

    @staticmethod
    def forward_transform(state, nodes, conns):
        N = nodes.shape[0]
        u_conns = jnp.zeros((N, N), dtype=jnp.float32)

        in_keys = jnp.asarray(conns[:, 0], jnp.int32)
        out_keys = jnp.asarray(conns[:, 1], jnp.int32)
        weights = conns[:, 2]

        u_conns = u_conns.at[in_keys, out_keys].set(weights)
        return nodes, u_conns

    @staticmethod
    def create_forward(config):
        act = Activation.name2func[config['h_activation']]
        agg = Aggregation.name2func[config['h_aggregation']]

        batch_act, batch_agg = vmap(act), vmap(agg)

        def forward(inputs, transform):

            inputs_with_bias = jnp.concatenate((inputs, jnp.ones((1,))), axis=0)
            nodes, weights = transform

            input_idx = config['h_input_idx']
            output_idx = config['h_output_idx']

            N = nodes.shape[0]
            vals = jnp.full((N,), 0.)

            def body_func(i, values):
                values = values.at[input_idx].set(inputs_with_bias)
                nodes_ins = values * weights.T
                values = batch_agg(nodes_ins)  # z = agg(ins)
                values = values * nodes[:, 2] + nodes[:, 1]  # z = z * response + bias
                values = batch_act(values)  # z = act(z)
                return values

            vals = jax.lax.fori_loop(0, config['h_activate_times'], body_func, vals)
            return vals[output_idx]

        return forward
