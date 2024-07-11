from typing import Union, Sequence, Callable, Optional

import numpy as np
import jax, jax.numpy as jnp
import sympy as sp
from tensorneat.common import (
    ACT,
    AGG,
    apply_activation,
    apply_aggregation,
    mutate_int,
    mutate_float,
    get_func_name
)

from . import BaseNode


class BiasNode(BaseNode):
    """
    Default node gene, with the same behavior as in NEAT-python.
    The attribute response is removed.
    """

    custom_attrs = ["bias", "aggregation", "activation"]

    def __init__(
        self,
        bias_init_mean: float = 0.0,
        bias_init_std: float = 1.0,
        bias_mutate_power: float = 0.15,
        bias_mutate_rate: float = 0.2,
        bias_replace_rate: float = 0.015,
        bias_lower_bound: float = -5,
        bias_upper_bound: float = 5,
        aggregation_default: Optional[Callable] = None,
        aggregation_options: Union[Callable, Sequence[Callable]] = AGG.sum,
        aggregation_replace_rate: float = 0.1,
        activation_default: Optional[Callable] = None,
        activation_options: Union[Callable, Sequence[Callable]] = ACT.sigmoid,
        activation_replace_rate: float = 0.1,
    ):
        super().__init__()

        if isinstance(aggregation_options, Callable):
            aggregation_options = [aggregation_options]
        if isinstance(activation_options, Callable):
            activation_options = [activation_options]

        if aggregation_default is None:
            aggregation_default = aggregation_options[0]
        if activation_default is None:
            activation_default = activation_options[0]

        self.bias_init_mean = bias_init_mean
        self.bias_init_std = bias_init_std
        self.bias_mutate_power = bias_mutate_power
        self.bias_mutate_rate = bias_mutate_rate
        self.bias_replace_rate = bias_replace_rate
        self.bias_lower_bound = bias_lower_bound
        self.bias_upper_bound = bias_upper_bound

        self.aggregation_default = aggregation_options.index(aggregation_default)
        self.aggregation_options = aggregation_options
        self.aggregation_indices = np.arange(len(aggregation_options))
        self.aggregation_replace_rate = aggregation_replace_rate

        self.activation_default = activation_options.index(activation_default)
        self.activation_options = activation_options
        self.activation_indices = np.arange(len(activation_options))
        self.activation_replace_rate = activation_replace_rate

    def new_identity_attrs(self, state):
        return jnp.array(
            [0, self.aggregation_default, -1]
        )  # activation=-1 means ACT.identity

    def new_random_attrs(self, state, randkey):
        k1, k2, k3 = jax.random.split(randkey, num=3)
        bias = jax.random.normal(k1, ()) * self.bias_init_std + self.bias_init_mean
        bias = jnp.clip(bias, self.bias_lower_bound, self.bias_upper_bound)
        agg = jax.random.choice(k2, self.aggregation_indices)
        act = jax.random.choice(k3, self.activation_indices)

        return jnp.array([bias, agg, act])

    def mutate(self, state, randkey, attrs):
        k1, k2, k3 = jax.random.split(randkey, num=3)
        bias, agg, act = attrs

        bias = mutate_float(
            k1,
            bias,
            self.bias_init_mean,
            self.bias_init_std,
            self.bias_mutate_power,
            self.bias_mutate_rate,
            self.bias_replace_rate,
        )
        bias = jnp.clip(bias, self.bias_lower_bound, self.bias_upper_bound)
        agg = mutate_int(
            k2, agg, self.aggregation_indices, self.aggregation_replace_rate
        )

        act = mutate_int(k3, act, self.activation_indices, self.activation_replace_rate)

        return jnp.array([bias, agg, act])

    def distance(self, state, attrs1, attrs2):
        bias1, agg1, act1 = attrs1
        bias2, agg2, act2 = attrs2

        return jnp.abs(bias1 - bias2) + (agg1 != agg2) + (act1 != act2)

    def forward(self, state, attrs, inputs, is_output_node=False):
        bias, agg, act = attrs

        z = apply_aggregation(agg, inputs, self.aggregation_options)
        z = bias + z

        # the last output node should not be activated
        z = jax.lax.cond(
            is_output_node, lambda: z, lambda: apply_activation(act, z, self.activation_options)
        )

        return z

    def repr(self, state, node, precision=2, idx_width=3, func_width=8):
        idx, bias, agg, act = node

        idx = int(idx)
        bias = round(float(bias), precision)
        agg = int(agg)
        act = int(act)

        if act == -1:
            act_func = ACT.identity
        else:
            act_func = self.activation_options[act]
        return "{}(idx={:<{idx_width}}, bias={:<{float_width}}, aggregation={:<{func_width}}, activation={:<{func_width}})".format(
            self.__class__.__name__,
            idx,
            bias,
            get_func_name(self.aggregation_options[agg]),
            get_func_name(act_func),
            idx_width=idx_width,
            float_width=precision + 3,
            func_width=func_width,
        )

    def to_dict(self, state, node):
        idx, bias, agg, act = node

        idx = int(idx)

        bias = jnp.float32(bias)
        agg = int(agg)
        act = int(act)

        if act == -1:
            act_func = ACT.identity
        else:
            act_func = self.activation_options[act]

        return {
            "idx": idx,
            "bias": bias,
            "agg": get_func_name(self.aggregation_options[agg]),
            "act": get_func_name(act_func),
        }

    def sympy_func(self, state, node_dict, inputs, is_output_node=False):
        bias = sp.symbols(f"n_{node_dict['idx']}_b")

        z = AGG.obtain_sympy(node_dict["agg"])(inputs)

        z = bias + z
        if is_output_node:
            pass
        else:
            z = ACT.obtain_sympy(node_dict["act"])(z)

        return z, {bias: node_dict["bias"]}
