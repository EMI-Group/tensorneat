from typing import Optional, Union, Sequence, Callable

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

from .base import BaseNode


class DefaultNode(BaseNode):
    "Default node gene, with the same behavior as in NEAT-python."

    custom_attrs = ["bias", "response", "aggregation", "activation"]

    def __init__(
        self,
        bias_init_mean: float = 0.0,
        bias_init_std: float = 1.0,
        bias_mutate_power: float = 0.15,
        bias_mutate_rate: float = 0.2,
        bias_replace_rate: float = 0.015,
        bias_lower_bound: float = -5,
        bias_upper_bound: float = 5,
        response_init_mean: float = 1.0,
        response_init_std: float = 0.0,
        response_mutate_power: float = 0.15,
        response_mutate_rate: float = 0.2,
        response_replace_rate: float = 0.015,
        response_lower_bound: float = -5,
        response_upper_bound: float = 5,
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

        self.response_init_mean = response_init_mean
        self.response_init_std = response_init_std
        self.response_mutate_power = response_mutate_power
        self.response_mutate_rate = response_mutate_rate
        self.response_replace_rate = response_replace_rate
        self.reponse_lower_bound = response_lower_bound
        self.response_upper_bound = response_upper_bound

        self.aggregation_default = aggregation_options.index(aggregation_default)
        self.aggregation_options = aggregation_options
        self.aggregation_indices = np.arange(len(aggregation_options))
        self.aggregation_replace_rate = aggregation_replace_rate

        self.activation_default = activation_options.index(activation_default)
        self.activation_options = activation_options
        self.activation_indices = np.arange(len(activation_options))
        self.activation_replace_rate = activation_replace_rate

    def new_identity_attrs(self, state):
        bias = 0
        res = 1
        agg = self.aggregation_default
        act = self.activation_default

        return jnp.array([bias, res, agg, act])  # activation=-1 means ACT.identity

    def new_random_attrs(self, state, randkey):
        k1, k2, k3, k4 = jax.random.split(randkey, num=4)
        bias = jax.random.normal(k1, ()) * self.bias_init_std + self.bias_init_mean
        bias = jnp.clip(bias, self.bias_lower_bound, self.bias_upper_bound)
        res = (
            jax.random.normal(k2, ()) * self.response_init_std + self.response_init_mean
        )
        res = jnp.clip(res, self.reponse_lower_bound, self.response_upper_bound)
        agg = jax.random.choice(k3, self.aggregation_indices)
        act = jax.random.choice(k4, self.activation_indices)

        return jnp.array([bias, res, agg, act])

    def mutate(self, state, randkey, attrs):
        k1, k2, k3, k4 = jax.random.split(randkey, num=4)
        bias, res, agg, act = attrs
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
        res = mutate_float(
            k2,
            res,
            self.response_init_mean,
            self.response_init_std,
            self.response_mutate_power,
            self.response_mutate_rate,
            self.response_replace_rate,
        )
        res = jnp.clip(res, self.reponse_lower_bound, self.response_upper_bound)
        agg = mutate_int(
            k4, agg, self.aggregation_indices, self.aggregation_replace_rate
        )

        act = mutate_int(k3, act, self.activation_indices, self.activation_replace_rate)

        return jnp.array([bias, res, agg, act])

    def distance(self, state, attrs1, attrs2):
        bias1, res1, agg1, act1 = attrs1
        bias2, res2, agg2, act2 = attrs2
        return (
            jnp.abs(bias1 - bias2)  # bias
            + jnp.abs(res1 - res2)  # response
            + (agg1 != agg2)  # aggregation
            + (act1 != act2)  # activation
        )

    def forward(self, state, attrs, inputs, is_output_node=False):
        bias, res, agg, act = attrs

        z = apply_aggregation(agg, inputs, self.aggregation_options)
        z = bias + res * z

        # the last output node should not be activated
        z = jax.lax.cond(
            is_output_node, lambda: z, lambda: apply_activation(act, z, self.activation_options)
        )

        return z

    def repr(self, state, node, precision=2, idx_width=3, func_width=8):
        idx, bias, res, agg, act = node

        idx = int(idx)
        bias = round(float(bias), precision)
        res = round(float(res), precision)
        agg = int(agg)
        act = int(act)

        if act == -1:
            act_func = ACT.identity
        else:
            act_func = self.activation_options[act]
        return "{}(idx={:<{idx_width}}, bias={:<{float_width}}, response={:<{float_width}}, aggregation={:<{func_width}}, activation={:<{func_width}})".format(
            self.__class__.__name__,
            idx,
            bias,
            res,
            get_func_name(self.aggregation_options[agg]),
            get_func_name(act_func),
            idx_width=idx_width,
            float_width=precision + 3,
            func_width=func_width,
        )

    def to_dict(self, state, node):
        idx, bias, res, agg, act = node

        idx = int(idx)
        bias = jnp.float32(bias)
        res = jnp.float32(res)
        agg = int(agg)
        act = int(act)

        if act == -1:
            act_func = ACT.identity
        else:
            act_func = self.activation_options[act]
        return {
            "idx": idx,
            "bias": bias,
            "res": res,
            "agg": get_func_name(self.aggregation_options[agg]),
            "act": get_func_name(act_func),
        }

    def sympy_func(self, state, node_dict, inputs, is_output_node=False):
        nd = node_dict
        bias = sp.symbols(f"n_{nd['idx']}_b")
        res = sp.symbols(f"n_{nd['idx']}_r")

        z = AGG.obtain_sympy(nd["agg"])(inputs)
        z = bias + res * z

        if is_output_node:
            pass
        else:
            z = ACT.obtain_sympy(nd["act"])(z)

        return z, {bias: nd["bias"], res: nd["res"]}
