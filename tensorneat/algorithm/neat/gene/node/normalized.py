from typing import Tuple

import jax, jax.numpy as jnp

from utils import Act, Agg, act, agg, mutate_int, mutate_float
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
        activation_default: callable = Act.sigmoid,
        activation_options: Tuple = (Act.sigmoid,),
        activation_replace_rate: float = 0.1,
        aggregation_default: callable = Agg.sum,
        aggregation_options: Tuple = (Agg.sum,),
        aggregation_replace_rate: float = 0.1,
        alpha_init_mean: float = 0.0,
        alpha_init_std: float = 1.0,
        alpha_mutate_power: float = 0.5,
        alpha_mutate_rate: float = 0.7,
        alpha_replace_rate: float = 0.1,
        beta_init_mean: float = 1.0,
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

        self.activation_default = activation_options.index(activation_default)
        self.activation_options = activation_options
        self.activation_indices = jnp.arange(len(activation_options))
        self.activation_replace_rate = activation_replace_rate

        self.aggregation_default = aggregation_options.index(aggregation_default)
        self.aggregation_options = aggregation_options
        self.aggregation_indices = jnp.arange(len(aggregation_options))
        self.aggregation_replace_rate = aggregation_replace_rate

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

    def new_custom_attrs(self, state):
        return jnp.array(
            [
                self.bias_init_mean,
                self.activation_default,
                self.aggregation_default,
                0,  # mean
                1,  # std
                self.alpha_init_mean,  # alpha
                self.beta_init_mean,  # beta
            ]
        )

    def new_random_attrs(self, state, randkey):
        k1, k2, k3, k4, k5, k6 = jax.random.split(randkey, num=6)
        bias = jax.random.normal(k1, ()) * self.bias_init_std + self.bias_init_mean
        act = jax.random.randint(k3, (), 0, len(self.activation_options))
        agg = jax.random.randint(k4, (), 0, len(self.aggregation_options))
        mean = 0
        std = 1
        alpha = jax.random.normal(k5, ()) * self.alpha_init_std + self.alpha_init_mean
        beta = jax.random.normal(k6, ()) * self.beta_init_std + self.beta_init_mean

        return jnp.array([bias, act, agg, 0, 1, alpha, beta])

    def mutate(self, state, randkey, node):
        k1, k2, k3, k4, k5, k6 = jax.random.split(state.randkey, num=6)
        index = node[0]

        bias = mutate_float(
            k1,
            node[1],
            self.bias_init_mean,
            self.bias_init_std,
            self.bias_mutate_power,
            self.bias_mutate_rate,
            self.bias_replace_rate,
        )

        act = mutate_int(
            k3, node[2], self.activation_indices, self.activation_replace_rate
        )

        agg = mutate_int(
            k4, node[3], self.aggregation_indices, self.aggregation_replace_rate
        )

        mean = node[4]
        std = node[5]

        alpha = mutate_float(
            k5,
            node[6],
            self.alpha_init_mean,
            self.alpha_init_std,
            self.alpha_mutate_power,
            self.alpha_mutate_rate,
            self.alpha_replace_rate,
        )

        beta = mutate_float(
            k6,
            node[7],
            self.beta_init_mean,
            self.beta_init_std,
            self.beta_mutate_power,
            self.beta_mutate_rate,
            self.beta_replace_rate,
        )

        return jnp.array([index, bias, act, agg, mean, std, alpha, beta])

    def distance(self, state, node1, node2):
        return (
            jnp.abs(node1[1] - node2[1])  # bias
            + (node1[2] != node2[2])  # activation
            + (node1[3] != node2[3])  # aggregation
            + (node1[6] - node2[6])  # alpha
            + (node1[7] - node2[7])  # beta
        )

    def forward(self, state, attrs, inputs, is_output_node=False):
        """
        post_act = (agg(inputs) + bias - mean) / std * alpha + beta
        """
        bias, act_idx, agg_idx, mean, std, alpha, beta = attrs

        z = agg(agg_idx, inputs, self.aggregation_options)
        z = bias + z
        z = (z - mean) / (std + self.eps) * alpha + beta  # normalization

        # the last output node should not be activated
        z = jax.lax.cond(
            is_output_node, lambda: z, lambda: act(act_idx, z, self.activation_options)
        )

        return z

    def update_by_batch(self, state, attrs, batch_inputs, is_output_node=False):

        bias, act_idx, agg_idx, mean, std, alpha, beta = attrs

        batch_z = jax.vmap(agg, in_axes=(None, 0, None))(
            agg_idx, batch_inputs, self.aggregation_options
        )

        batch_z = bias + batch_z

        # calculate mean
        valid_values_count = jnp.sum(~jnp.isnan(batch_inputs))
        valid_values_sum = jnp.sum(jnp.where(jnp.isnan(batch_inputs), 0, batch_inputs))
        mean = valid_values_sum / valid_values_count

        # calculate std
        std = jnp.sqrt(
            jnp.sum(jnp.where(jnp.isnan(batch_inputs), 0, (batch_inputs - mean) ** 2))
            / valid_values_count
        )

        batch_z = (batch_z - mean) / (std + self.eps) * alpha + beta  # normalization
        batch_z = jax.lax.cond(
            is_output_node,
            lambda: batch_z,
            lambda: jax.vmap(act, in_axes=(None, 0, None))(
                act_idx, batch_z, self.activation_options
            ),
        )

        # update mean and std to the attrs
        attrs = attrs.at[3].set(mean)
        attrs = attrs.at[4].set(std)

        return batch_z, attrs
