from typing import Tuple, Union, Sequence, Callable

import numpy as np
import jax, jax.numpy as jnp
import sympy as sp

from tensorneat.common import (
    Act,
    Agg,
    act_func,
    agg_func,
    mutate_int,
    mutate_float,
    convert_to_sympy,
)

from . import BaseNodeGene


class DefaultNodeGene(BaseNodeGene):
    "Default node gene, with the same behavior as in NEAT-python."

    custom_attrs = ["bias", "response", "aggregation", "activation"]

    def __init__(
        self,
        bias_init_mean: float = 0.0,
        bias_init_std: float = 1.0,
        bias_mutate_power: float = 0.5,
        bias_mutate_rate: float = 0.7,
        bias_replace_rate: float = 0.1,
        response_init_mean: float = 1.0,
        response_init_std: float = 0.0,
        response_mutate_power: float = 0.5,
        response_mutate_rate: float = 0.7,
        response_replace_rate: float = 0.1,
        aggregation_default: Callable = Agg.sum,
        aggregation_options: Union[Callable, Sequence[Callable]] = Agg.sum,
        aggregation_replace_rate: float = 0.1,
        activation_default: Callable = Act.sigmoid,
        activation_options: Union[Callable, Sequence[Callable]] = Act.sigmoid,
        activation_replace_rate: float = 0.1,
    ):
        super().__init__()

        if isinstance(aggregation_options, Callable):
            aggregation_options = [aggregation_options]
        if isinstance(activation_options, Callable):
            activation_options = [activation_options]

        self.bias_init_mean = bias_init_mean
        self.bias_init_std = bias_init_std
        self.bias_mutate_power = bias_mutate_power
        self.bias_mutate_rate = bias_mutate_rate
        self.bias_replace_rate = bias_replace_rate

        self.response_init_mean = response_init_mean
        self.response_init_std = response_init_std
        self.response_mutate_power = response_mutate_power
        self.response_mutate_rate = response_mutate_rate
        self.response_replace_rate = response_replace_rate

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
            [0, 1, self.aggregation_default, -1]
        )  # activation=-1 means Act.identity

    def new_random_attrs(self, state, randkey):
        k1, k2, k3, k4 = jax.random.split(randkey, num=4)
        bias = jax.random.normal(k1, ()) * self.bias_init_std + self.bias_init_mean
        res = (
            jax.random.normal(k2, ()) * self.response_init_std + self.response_init_mean
        )
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

        res = mutate_float(
            k2,
            res,
            self.response_init_mean,
            self.response_init_std,
            self.response_mutate_power,
            self.response_mutate_rate,
            self.response_replace_rate,
        )

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

        z = agg_func(agg, inputs, self.aggregation_options)
        z = bias + res * z

        # the last output node should not be activated
        z = jax.lax.cond(
            is_output_node, lambda: z, lambda: act_func(act, z, self.activation_options)
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
            act_func = Act.identity
        else:
            act_func = self.activation_options[act]
        return "{}(idx={:<{idx_width}}, bias={:<{float_width}}, response={:<{float_width}}, aggregation={:<{func_width}}, activation={:<{func_width}})".format(
            self.__class__.__name__,
            idx,
            bias,
            res,
            self.aggregation_options[agg].__name__,
            act_func.__name__,
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
            act_func = Act.identity
        else:
            act_func = self.activation_options[act]
        return {
            "idx": idx,
            "bias": bias,
            "res": res,
            "agg": self.aggregation_options[int(agg)].__name__,
            "act": act_func.__name__,
        }

    def sympy_func(self, state, node_dict, inputs, is_output_node=False):
        nd = node_dict
        bias = sp.symbols(f"n_{nd['idx']}_b")
        res = sp.symbols(f"n_{nd['idx']}_r")

        z = convert_to_sympy(nd["agg"])(inputs)
        z = bias + res * z

        if is_output_node:
            pass
        else:
            z = convert_to_sympy(nd["act"])(z)

        return z, {bias: nd["bias"], res: nd["res"]}
