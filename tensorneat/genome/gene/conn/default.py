import jax.numpy as jnp
import jax.random
import sympy as sp
from tensorneat.common import mutate_float
from .base import BaseConn


class DefaultConn(BaseConn):
    "Default connection gene, with the same behavior as in NEAT-python."

    custom_attrs = ["weight"]

    def __init__(
        self,
        weight_init_mean: float = 0.0,
        weight_init_std: float = 1.0,
        weight_mutate_power: float = 0.15,
        weight_mutate_rate: float = 0.2,
        weight_replace_rate: float = 0.015,
        weight_lower_bound: float = -5.0,
        weight_upper_bound: float = 5.0,
    ):
        super().__init__()
        self.weight_init_mean = weight_init_mean
        self.weight_init_std = weight_init_std
        self.weight_mutate_power = weight_mutate_power
        self.weight_mutate_rate = weight_mutate_rate
        self.weight_replace_rate = weight_replace_rate
        self.weight_lower_bound = weight_lower_bound
        self.weight_upper_bound = weight_upper_bound


    def new_zero_attrs(self, state):
        return jnp.array([0.0])  # weight = 0

    def new_identity_attrs(self, state):
        return jnp.array([1.0])  # weight = 1

    def new_random_attrs(self, state, randkey):
        weight = (
            jax.random.normal(randkey, ()) * self.weight_init_std
            + self.weight_init_mean
        )
        weight = jnp.clip(weight, self.weight_lower_bound, self.weight_upper_bound)
        return jnp.array([weight])

    def mutate(self, state, randkey, attrs):
        weight = attrs[0]
        weight = mutate_float(
            randkey,
            weight,
            self.weight_init_mean,
            self.weight_init_std,
            self.weight_mutate_power,
            self.weight_mutate_rate,
            self.weight_replace_rate,
        )
        weight = jnp.clip(weight, self.weight_lower_bound, self.weight_upper_bound)
        return jnp.array([weight])

    def distance(self, state, attrs1, attrs2):
        weight1 = attrs1[0]
        weight2 = attrs2[0]
        return jnp.abs(weight1 - weight2)

    def forward(self, state, attrs, inputs):
        weight = attrs[0]
        return inputs * weight

    def repr(self, state, conn, precision=2, idx_width=3, func_width=8):
        in_idx, out_idx, weight = conn

        in_idx = int(in_idx)
        out_idx = int(out_idx)
        weight = round(float(weight), precision)

        return "{}(in: {:<{idx_width}}, out: {:<{idx_width}}, weight: {:<{float_width}})".format(
            self.__class__.__name__,
            in_idx,
            out_idx,
            weight,
            idx_width=idx_width,
            float_width=precision + 3,
        )

    def to_dict(self, state, conn):
        return {
            "in": int(conn[0]),
            "out": int(conn[1]),
            "weight": jnp.float32(conn[2]),
        }

    def sympy_func(self, state, conn_dict, inputs, precision=None):
        weight = sp.symbols(f"c_{conn_dict['in']}_{conn_dict['out']}_w")

        return inputs * weight, {weight: conn_dict["weight"]}
