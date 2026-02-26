from turtle import forward

import jax, jax.numpy as jnp

from tensorneat.common import (
    ACT,
    AGG,
    mutate_float,
    apply_aggregation,
    apply_activation
)
from .base import BaseNode


class RNNNode(BaseNode):
    custom_attrs = ["i_to_h_response", "hm1_to_h_response", "h_to_o_response", "h_bias", "o_bias"]

    def __init__(
        self,
        bias_init_mean: float = 0.0,
        bias_init_std: float = 0.2,
        bias_mutate_power: float = 0.15,
        bias_mutate_rate: float = 0.2,
        bias_replace_rate: float = 0.015,
        bias_lower_bound: float = -1,
        bias_upper_bound: float = 1,
        response_init_mean: float = 0.0,
        response_init_std: float = 0.2,
        response_mutate_power: float = 0.15,
        response_mutate_rate: float = 0.2,
        response_replace_rate: float = 0.015,
        response_lower_bound: float = -1,
        response_upper_bound: float = 1,
    ):
        super().__init__()
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
        self.response_lower_bound = response_lower_bound
        self.response_upper_bound = response_upper_bound

    def new_identity_attrs(self, state):
        i_to_h_res = 1; hm1_to_h_res = 0; h_to_o_res = 1
        h_bias = 0; o_bias = 0
        return jnp.array([i_to_h_res, hm1_to_h_res, h_to_o_res, h_bias, o_bias])
    
    def new_random_attrs(self, state, randkey):
        k1, k2, k3, k4, k5 = jax.random.split(randkey, num=5)
        # new random attrs
        i_to_h_res = jax.random.normal(k1, ()) * self.response_init_std + self.response_init_mean
        hm1_to_h_res = jax.random.normal(k2, ()) * self.response_init_std + self.response_init_mean
        h_to_o_res = jax.random.normal(k3, ()) * self.response_init_std + self.response_init_mean

        h_bias = jax.random.normal(k4, ()) * self.bias_init_std + self.bias_init_mean
        o_bias = jax.random.normal(k5, ()) * self.bias_init_std + self.bias_init_mean
        # clipping
        i_to_h_res = jnp.clip(i_to_h_res, self.response_lower_bound, self.response_upper_bound)
        hm1_to_h_res = jnp.clip(hm1_to_h_res, self.response_lower_bound, self.response_upper_bound)
        h_to_o_res = jnp.clip(h_to_o_res, self.response_lower_bound, self.response_upper_bound)

        h_bias = jnp.clip(h_bias, self.bias_lower_bound, self.bias_upper_bound)
        o_bias = jnp.clip(o_bias, self.bias_lower_bound, self.bias_upper_bound)

        return jnp.array([i_to_h_res, hm1_to_h_res, h_to_o_res, h_bias, o_bias])
    
    def mutate(self, state, randkey, attrs):
        keys = jax.random.split(randkey, num=5)
        attrs = jnp.array(attrs)
        responses = jax.vmap(mutate_float, in_axes=(0, 0, None, None, None, None, None))(
            keys[:3],
            attrs[:3],
            self.response_init_mean,
            self.response_init_std,
            self.response_mutate_power,
            self.response_mutate_rate,
            self.response_replace_rate
        )
        responses = jnp.clip(responses, self.response_lower_bound, self.response_upper_bound)
        biases = jax.vmap(mutate_float, in_axes=(0, 0, None, None, None, None, None))(
            keys[3:],
            attrs[3:],
            self.bias_init_mean,
            self.bias_init_std,
            self.bias_mutate_power,
            self.bias_mutate_rate,
            self.bias_replace_rate
        )
        biases = jnp.clip(biases, self.bias_lower_bound, self.bias_upper_bound)
        return jnp.concat((responses, biases))
    
    def distance(self, state, attrs1, attrs2):
        return jnp.abs(attrs1 - attrs2).sum()
    
    def forward(self, state, attrs, inputs, is_output_node=False):
        i_to_h_res, hm1_to_h_res, h_to_o_res, h_bias, o_bias = attrs
        hm1 = inputs[0] # extract last_hidden_state
        x = inputs[1:] # extract inputs from this rollout_step
        i = jnp.sum(x) # only sum for now
        h = jnp.tanh(i_to_h_res * i + hm1_to_h_res * hm1 + h_bias)
        o = h_to_o_res * h + o_bias
        return o, h
