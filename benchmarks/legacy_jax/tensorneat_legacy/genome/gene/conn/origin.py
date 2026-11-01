import jax, jax.numpy as jnp
from .default import DefaultConn


class OriginConn(DefaultConn):
    """
    Implementation of connections in origin NEAT Paper.
    Details at https://github.com/EMI-Group/tensorneat/issues/11.
    """

    # add historical_marker into fixed_attrs
    fixed_attrs = ["input_index", "output_index", "historical_marker"]
    custom_attrs = ["weight"]

    def __init__(
        self,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

    def crossover(self, state, randkey, attrs1, attrs2):
        # random pick one of attrs, without attrs exchange
        return jnp.where(
            # origin code, generate multiple random numbers, without attrs exchange
            # jax.random.normal(randkey, attrs1.shape) > 0,
            jax.random.normal(randkey)
            > 0,  # generate one random number, without attrs exchange
            attrs1,
            attrs2,
        )

    def get_historical_marker(self, state, gene_array):
        return gene_array[2]
    
    def repr(self, state, conn, precision=2, idx_width=3, func_width=8):
        in_idx, out_idx, historical_marker, weight = conn

        in_idx = int(in_idx)
        out_idx = int(out_idx)
        historical_marker = int(historical_marker)
        weight = round(float(weight), precision)

        return "{}(in: {:<{idx_width}}, out: {:<{idx_width}}, historical_marker: {:<{idx_width}}, weight: {:<{float_width}})".format(
            self.__class__.__name__,
            in_idx,
            out_idx,
            historical_marker,
            weight,
            idx_width=idx_width,
            float_width=precision + 3,
        )

    def to_dict(self, state, conn):
        return {
            "in": int(conn[0]),
            "out": int(conn[1]),
            "historical_marker": int(conn[2]),
            "weight": jnp.float32(conn[3]),
        }