from typing import Tuple

import jax, jax.numpy as jnp

from tensorneat.common import Act, Agg, act_func, agg_func, mutate_int, mutate_float
from . import BaseNodeGene


class NormalizedNode(BaseNodeGene):
    """
    Node with normalization before activation.
    """

    # alpha and beta is used for normalization, just like BatchNorm
    # norm: (data - mean) / (std + eps) * alpha + beta
    custom_attrs = ["bias", "aggregation", "activation", "mean", "std", "alpha", "beta"]
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
        alpha_init_mean: float = 1.0,
        alpha_init_std: float = 1.0,
        alpha_mutate_power: float = 0.5,
        alpha_mutate_rate: float = 0.7,
        alpha_replace_rate: float = 0.1,
        beta_init_mean: float = 0.0,
        beta_init_std: float = 1.0,
        beta_mutate_power: float = 0.5,
        beta_mutate_rate: float = 0.7,
        beta_replace_rate: float = 0.1,
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

        self.alpha_init_mean = alpha_init_mean
        self.alpha_init_std = alpha_init_std
        self.alpha_mutate_power = alpha_mutate_power
        self.alpha_mutate_rate = alpha_mutate_rate
        self.alpha_replace_rate = alpha_replace_rate

        self.beta_init_mean = beta_init_mean
        self.beta_init_std = beta_init_std
        self.beta_mutate_power = beta_mutate_power
        self.beta_mutate_rate = beta_mutate_rate
        self.beta_replace_rate = beta_replace_rate

    def new_identity_attrs(self, state):
        return jnp.array(
            [0, self.aggregation_default, -1, 0, 1, 1, 0]
        )  # activation=-1 means Act.identity

    def new_random_attrs(self, state, randkey):
        k1, k2, k3, k4, k5 = jax.random.split(randkey, num=5)
        bias = jax.random.normal(k1, ()) * self.bias_init_std + self.bias_init_mean
        agg = jax.random.randint(k2, (), 0, len(self.aggregation_options))
        act = jax.random.randint(k3, (), 0, len(self.activation_options))

        mean = 0
        std = 1
        alpha = jax.random.normal(k4, ()) * self.alpha_init_std + self.alpha_init_mean
        beta = jax.random.normal(k5, ()) * self.beta_init_std + self.beta_init_mean

        return jnp.array([bias, agg, act, mean, std, alpha, beta])

    def mutate(self, state, randkey, attrs):
        k1, k2, k3, k4, k5 = jax.random.split(randkey, num=5)
        bias, act, agg, mean, std, alpha, beta = attrs

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

        alpha = mutate_float(
            k4,
            alpha,
            self.alpha_init_mean,
            self.alpha_init_std,
            self.alpha_mutate_power,
            self.alpha_mutate_rate,
            self.alpha_replace_rate,
        )

        beta = mutate_float(
            k5,
            beta,
            self.beta_init_mean,
            self.beta_init_std,
            self.beta_mutate_power,
            self.beta_mutate_rate,
            self.beta_replace_rate,
        )

        return jnp.array([bias, agg, act, mean, std, alpha, beta])

    def distance(self, state, attrs1, attrs2):
        bias1, agg1, act1, mean1, std1, alpha1, beta1 = attrs1
        bias2, agg2, act2, mean2, std2, alpha2, beta2 = attrs2
        return (
            jnp.abs(bias1 - bias2)  # bias
            + (agg1 != agg2)  # aggregation
            + (act1 != act2)  # activation
            + jnp.abs(alpha1 - alpha2)  # alpha
            + jnp.abs(beta1 - beta2)  # beta
        )

    def forward(self, state, attrs, inputs, is_output_node=False):
        """
        post_act = (agg(inputs) + bias - mean) / std * alpha + beta
        """
        bias, agg, act, mean, std, alpha, beta = attrs

        z = agg_func(agg, inputs, self.aggregation_options)
        z = bias + z
        z = (z - mean) / (std + self.eps) * alpha + beta  # normalization

        # the last output node should not be activated
        z = jax.lax.cond(
            is_output_node, lambda: z, lambda: act_func(act, z, self.activation_options)
        )

        return z

    def input_transform(self, state, attrs, inputs):
        """
        make transform in the input node.
        the normalization also need be done in the first node.
        """
        bias, agg, act, mean, std, alpha, beta = attrs
        inputs = (inputs - mean) / (std + self.eps) * alpha + beta  # normalization
        return inputs

    def update_by_batch(self, state, attrs, batch_inputs, is_output_node=False):

        bias, agg, act, mean, std, alpha, beta = attrs

        batch_z = jax.vmap(agg_func, in_axes=(None, 0, None))(
            agg, batch_inputs, self.aggregation_options
        )

        batch_z = bias + batch_z

        # calculate mean
        valid_values_count = jnp.sum(~jnp.isnan(batch_z))
        valid_values_sum = jnp.sum(jnp.where(jnp.isnan(batch_z), 0, batch_z))
        mean = valid_values_sum / valid_values_count

        # calculate std
        std = jnp.sqrt(
            jnp.sum(jnp.where(jnp.isnan(batch_z), 0, (batch_z - mean) ** 2))
            / valid_values_count
        )

        batch_z = (batch_z - mean) / (std + self.eps) * alpha + beta  # normalization
        batch_z = jax.lax.cond(
            is_output_node,
            lambda: batch_z,
            lambda: jax.vmap(act_func, in_axes=(None, 0, None))(
                act, batch_z, self.activation_options
            ),
        )

        # update mean and std to the attrs
        attrs = attrs.at[3].set(mean)
        attrs = attrs.at[4].set(std)

        return batch_z, attrs

    def update_input_transform(self, state, attrs, batch_inputs):
        """
        update the attrs for transformation in the input node.
        default: do nothing
        """
        bias, agg, act, mean, std, alpha, beta = attrs

        # calculate mean
        valid_values_count = jnp.sum(~jnp.isnan(batch_inputs))
        valid_values_sum = jnp.sum(jnp.where(jnp.isnan(batch_inputs), 0, batch_inputs))
        mean = valid_values_sum / valid_values_count

        # calculate std
        std = jnp.sqrt(
            jnp.sum(jnp.where(jnp.isnan(batch_inputs), 0, (batch_inputs - mean) ** 2))
            / valid_values_count
        )

        batch_inputs = (batch_inputs - mean) / (
            std + self.eps
        ) * alpha + beta  # normalization

        # update mean and std to the attrs
        attrs = attrs.at[3].set(mean)
        attrs = attrs.at[4].set(std)

        return batch_inputs, attrs
