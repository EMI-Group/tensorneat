import jax, jax.numpy as jnp

from tensorneat.common import (
    ACT,
    AGG,
    mutate_float,
    apply_aggregation,
    apply_activation,
)
from tensorneat.common.functions import sigmoid_, sum_
from .base import BaseNode


class RNNNode(BaseNode):
    custom_attrs = ["bias", "response", "rnn_active", "retention_rate"]

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
        rnn_switching_rate: float = 0.001,
        retention_init_mean: float = 0.0,
        retention_init_std: float = 0.0,
        retention_mutate_power: float = 0.15,
        retention_mutate_rate: float = 0.2,
        retention_replace_rate: float = 0.015,
        retention_lower_bound: float = -5,
        retention_upper_bound: float = 5,
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
        self.rnn_switching_rate = rnn_switching_rate

        self.retention_init_mean = retention_init_mean
        self.retention_init_std = retention_init_std
        self.retention_mutate_power = retention_mutate_power
        self.retention_mutate_rate = retention_mutate_rate
        self.retention_replace_rate = retention_replace_rate
        self.retention_lower_bound = retention_lower_bound
        self.retention_upper_bound = retention_upper_bound

    def new_identity_attrs(self, state):
        bias=0
        response=1
        rnn_active=0
        retention_rate=0
        return jnp.array([bias, response, rnn_active, retention_rate])
    
    def new_random_attrs(self, state, randkey):
        k1, k2, k3 = jax.random.split(randkey, num=3)
        # new random attrs
        bias = jax.random.normal(k1, ()) * self.bias_init_std + self.bias_init_mean
        bias = jnp.clip(bias, self.bias_lower_bound, self.bias_upper_bound)

        response = jax.random.normal(k2, ()) * self.response_init_std + self.response_init_mean
        response = jnp.clip(response, self.response_lower_bound, self.response_upper_bound)

        rnn_active = 0
        # clipping
        retention_rate = jax.random.normal(k3, ()) * self.retention_init_std + self.retention_init_mean
        retention_rate = jnp.clip(retention_rate, self.retention_lower_bound, self.retention_upper_bound)

        return jnp.array([bias, response, rnn_active, retention_rate])
    
    def mutate(self, state, randkey, attrs):
        k1, k2, k3, k4 = jax.random.split(randkey, num=4)
        bias, response, _, retention_rate = attrs
        bias = mutate_float(
            k1,
            bias,
            self.bias_init_mean,
            self.bias_init_std,
            self.bias_mutate_power,
            self.bias_mutate_rate,
            self.bias_replace_rate
        )
        bias = jnp.clip(bias, self.bias_lower_bound, self.bias_upper_bound)
        response = mutate_float(
            k2,
            response,
            self.response_init_mean,
            self.response_init_std,
            self.response_mutate_power,
            self.response_mutate_rate,
            self.response_replace_rate
        )
        response = jnp.clip(response, self.response_lower_bound, self.response_upper_bound)
        retention_rate = mutate_float(
            k3,
            retention_rate,
            self.retention_init_mean,
            self.retention_init_std,
            self.retention_mutate_power,
            self.retention_mutate_rate,
            self.retention_replace_rate
        )
        retention_rate = jnp.clip(retention_rate, self.retention_lower_bound, self.retention_upper_bound)
        rnn_active = jax.random.choice(k4, 2, p=jnp.array([1-self.rnn_switching_rate, self.rnn_switching_rate]))

        return jnp.array([bias, response, rnn_active, retention_rate])
    
    def distance(self, state, attrs1, attrs2):
        bias1, response1, rnn_active1, retention_rate1 = attrs1
        bias2, response2, rnn_active2, retention_rate2 = attrs2
        return (
            jnp.abs(bias1 - bias2)
            + jnp.abs(response1 - response2)
            + (rnn_active1 != rnn_active2) * jnp.abs(retention_rate1 - retention_rate2)
        )
    

    def forward(self, state, attrs, inputs, h_old, is_output_node=False):
        bias, response, rnn_active, retention_rate = attrs
        x = sum_(inputs) # extract inputs from this rollout_step - sum only for now
        unclipped_h_new = x * response + bias + rnn_active * h_old * retention_rate
        h_new = jnp.tanh(unclipped_h_new)
        out = jax.lax.cond(is_output_node, lambda: unclipped_h_new, lambda: h_new)
        return out, h_new # output, next_hidden_state
