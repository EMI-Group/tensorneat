import jax
from jax import Array, numpy as jnp, vmap

from .normal import NormalGene
from .activation import Activation
from .aggregation import Aggregation
from algorithm.utils import unflatten_connections


class RecurrentGene(NormalGene):

    @staticmethod
    def forward_transform(state, nodes, conns):
        u_conns = unflatten_connections(nodes, conns)

        # remove un-enable connections and remove enable attr
        conn_enable = jnp.where(~jnp.isnan(u_conns[0]), True, False)
        u_conns = jnp.where(conn_enable, u_conns[1:, :], jnp.nan)

        return nodes, u_conns

    @staticmethod
    def create_forward(config):
        config['activation_funcs'] = [Activation.name2func[name] for name in config['activation_option_names']]
        config['aggregation_funcs'] = [Aggregation.name2func[name] for name in config['aggregation_option_names']]

        def act(idx, z):
            """
            calculate activation function for each node
            """
            idx = jnp.asarray(idx, dtype=jnp.int32)
            # change idx from float to int
            res = jax.lax.switch(idx, config['activation_funcs'], z)
            return res

        def agg(idx, z):
            """
            calculate activation function for inputs of node
            """
            idx = jnp.asarray(idx, dtype=jnp.int32)

            def all_nan():
                return 0.

            def not_all_nan():
                return jax.lax.switch(idx, config['aggregation_funcs'], z)

            return jax.lax.cond(jnp.all(jnp.isnan(z)), all_nan, not_all_nan)

        batch_act, batch_agg = vmap(act), vmap(agg)

        def forward(inputs, transform) -> Array:
            """
            jax forward for single input shaped (input_num, )
            nodes, connections are a single genome

            :argument inputs: (input_num, )
            :argument cal_seqs: (N, )
            :argument nodes: (N, 5)
            :argument connections: (2, N, N)

            :return (output_num, )
            """

            nodes, cons = transform

            input_idx = config['input_idx']
            output_idx = config['output_idx']

            N = nodes.shape[0]
            vals = jnp.full((N,), 0.)

            weights = cons[0, :]

            def body_func(i, values):
                values = values.at[input_idx].set(inputs)
                nodes_ins = values * weights.T
                values = batch_agg(nodes[:, 4], nodes_ins)  # z = agg(ins)
                values = values * nodes[:, 2] + nodes[:, 1]  # z = z * response + bias
                values = batch_act(nodes[:, 3], values)  # z = act(z)
                return values

            # for i in range(config['activate_times']):
            #     vals = body_func(i, vals)
            #
            # return vals[output_idx]
            vals = jax.lax.fori_loop(0, config['activate_times'], body_func, vals)
            return vals[output_idx]

        return forward
