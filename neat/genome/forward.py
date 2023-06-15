import jax
from jax import Array, numpy as jnp
from jax import jit, vmap

from .aggregations import agg
from .activations import act
from .utils import I_INT

# TODO: enabled information doesn't influence forward. That is wrong!
@jit
def forward_single(inputs: Array, cal_seqs: Array, nodes: Array, connections: Array,
                   input_idx: Array, output_idx: Array) -> Array:
    """
    jax forward for single input shaped (input_num, )
    nodes, connections are single genome

    :argument inputs: (input_num, )
    :argument input_idx: (input_num, )
    :argument output_idx: (output_num, )
    :argument cal_seqs: (N, )
    :argument nodes: (N, 5)
    :argument connections: (2, N, N)

    :return (output_num, )
    """
    N = nodes.shape[0]
    ini_vals = jnp.full((N,), jnp.nan)
    ini_vals = ini_vals.at[input_idx].set(inputs)

    def scan_body(carry, i):
        def hit():
            ins = carry * connections[0, :, i]
            z = agg(nodes[i, 4], ins)
            z = z * nodes[i, 2] + nodes[i, 1]
            z = act(nodes[i, 3], z)

            new_vals = carry.at[i].set(z)
            return new_vals

        def miss():
            return carry

        return jax.lax.cond((i == I_INT) | (jnp.isin(i, input_idx)), miss, hit), None

    vals, _ = jax.lax.scan(scan_body, ini_vals, cal_seqs)

    return vals[output_idx]
