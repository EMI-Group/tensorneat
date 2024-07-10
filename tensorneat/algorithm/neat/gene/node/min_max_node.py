from typing import Tuple

import jax, jax.numpy as jnp

from tensorneat.common import Act, Agg, act_func, agg_func, mutate_int, mutate_float
from . import BaseNodeGene


class MinMaxNode(BaseNodeGene):
    """
    Node with normalization before activation.
    """

    # alpha and beta is used for normalization, just like BatchNorm
    # norm: z = act(agg(inputs) + bias)
    #       z = (z - min) / (max - min) * (max_out - min_out) + min_out
    custom_attrs = ["bias", "aggregation", "activation", "min", "max"]
    eps = 1e-6

    def __init__(
        self,
        bias_init_mean: float = 0.0,
        bias_init_std: float = 1.0,
        bias_mutate_power: float = 0.5,
        bias_mutate_rate: float = 0.7,
        bias_replace_rate: float = 0.1,
        aggregation_default: callable = Agg.sum,
        aggregation_options: Tuple = (Agg.sum,),
        aggregation_replace_rate: float = 0.1,
        activation_default: callable = Act.sigmoid,
        activation_options: Tuple = (Act.sigmoid,),
        activation_replace_rate: float = 0.1,
        output_range: Tuple[float, float] = (-1, 1),
        update_hidden_node: bool = False,
    ):
        super().__init__()
        self.bias_init_mean = bias_init_mean
        self.bias_init_std = bias_init_std
        self.bias_mutate_power = bias_mutate_power
        self.bias_mutate_rate = bias_mutate_rate
        self.bias_replace_rate = bias_replace_rate

        self.aggregation_default = aggregation_options.index(aggregation_default)
        self.aggregation_options = aggregation_options
        self.aggregation_indices = jnp.arange(len(aggregation_options))
        self.aggregation_replace_rate = aggregation_replace_rate

        self.activation_default = activation_options.index(activation_default)
        self.activation_options = activation_options
        self.activation_indices = jnp.arange(len(activation_options))
        self.activation_replace_rate = activation_replace_rate

        self.output_range = output_range
        assert (
            len(self.output_range) == 2 and self.output_range[0] < self.output_range[1]
        )
        self.update_hidden_node = update_hidden_node

    def new_identity_attrs(self, state):
        return jnp.array(
            [0, self.aggregation_default, -1, 0, 1]
        )  # activation=-1 means Act.identity; min=0, max=1 will do not influence

    def new_random_attrs(self, state, randkey):
        k1, k2, k3, k4, k5 = jax.random.split(randkey, num=5)
        bias = jax.random.normal(k1, ()) * self.bias_init_std + self.bias_init_mean
        agg = jax.random.randint(k2, (), 0, len(self.aggregation_options))
        act = jax.random.randint(k3, (), 0, len(self.activation_options))
        return jnp.array([bias, agg, act, 0, 1])

    def mutate(self, state, randkey, attrs):
        k1, k2, k3, k4, k5 = jax.random.split(randkey, num=5)
        bias, act, agg, min_, max_ = attrs

        bias = mutate_float(
            k1,
            bias,
            self.bias_init_mean,
            self.bias_init_std,
            self.bias_mutate_power,
            self.bias_mutate_rate,
            self.bias_replace_rate,
        )

        agg = mutate_int(
            k2, agg, self.aggregation_indices, self.aggregation_replace_rate
        )

        act = mutate_int(k3, act, self.activation_indices, self.activation_replace_rate)

        return jnp.array([bias, agg, act, min_, max_])

    def distance(self, state, attrs1, attrs2):
        bias1, agg1, act1, min1, max1 = attrs1
        bias2, agg2, act2, min1, max1 = attrs2
        return (
            jnp.abs(bias1 - bias2)  # bias
            + (agg1 != agg2)  # aggregation
            + (act1 != act2)  # activation
        )

    def forward(self, state, attrs, inputs, is_output_node=False):
        """
        post_act = (agg(inputs) + bias - mean) / std * alpha + beta
        """
        bias, agg, act, min_, max_ = attrs

        z = agg_func(agg, inputs, self.aggregation_options)
        z = bias + z

        # the last output node should not be activated
        z = jax.lax.cond(
            is_output_node, lambda: z, lambda: act_func(act, z, self.activation_options)
        )

        if self.update_hidden_node:
            z = (z - min_) / (max_ - min_)  # transform to 01
            z = (
                z * (self.output_range[1] - self.output_range[0]) + self.output_range[0]
            )  # transform to output_range

        return z

    def input_transform(self, state, attrs, inputs):
        """
        make transform in the input node.
        the normalization also need be done in the first node.
        """
        bias, agg, act, min_, max_ = attrs
        inputs = (inputs - min_) / (max_ - min_)  # transform to 01
        inputs = (
            inputs * (self.output_range[1] - self.output_range[0])
            + self.output_range[0]
        )
        return inputs

    def update_by_batch(self, state, attrs, batch_inputs, is_output_node=False):

        bias, agg, act, min_, max_ = attrs

        batch_z = jax.vmap(agg_func, in_axes=(None, 0, None))(
            agg, batch_inputs, self.aggregation_options
        )

        batch_z = bias + batch_z

        batch_z = jax.lax.cond(
            is_output_node,
            lambda: batch_z,
            lambda: jax.vmap(act_func, in_axes=(None, 0, None))(
                act, batch_z, self.activation_options
            ),
        )

        if self.update_hidden_node:
            # calculate min, max
            min_ = jnp.min(jnp.where(jnp.isnan(batch_z), jnp.inf, batch_z))
            max_ = jnp.max(jnp.where(jnp.isnan(batch_z), -jnp.inf, batch_z))

            batch_z = (batch_z - min_) / (max_ - min_)  # transform to 01
            batch_z = (
                batch_z * (self.output_range[1] - self.output_range[0])
                + self.output_range[0]
            )

            # update mean and std to the attrs
            attrs = attrs.at[3].set(min_)
            attrs = attrs.at[4].set(max_)

        return batch_z, attrs

    def update_input_transform(self, state, attrs, batch_inputs):
        """
        update the attrs for transformation in the input node.
        default: do nothing
        """
        bias, agg, act, min_, max_ = attrs

        # calculate min, max
        min_ = jnp.min(jnp.where(jnp.isnan(batch_inputs), jnp.inf, batch_inputs))
        max_ = jnp.max(jnp.where(jnp.isnan(batch_inputs), -jnp.inf, batch_inputs))

        batch_inputs = (batch_inputs - min_) / (max_ - min_)  # transform to 01
        batch_inputs = (
            batch_inputs * (self.output_range[1] - self.output_range[0])
            + self.output_range[0]
        )

        # update mean and std to the attrs
        attrs = attrs.at[3].set(min_)
        attrs = attrs.at[4].set(max_)

        return batch_inputs, attrs
