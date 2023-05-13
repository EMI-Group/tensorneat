"""
Lowers, compiles, and creates functions used in the NEAT pipeline.
"""
from functools import partial
import time

import numpy as np
from jax import jit, vmap

from .genome import act_name2key, agg_name2key, initialize_genomes
from .genome import topological_sort, forward_single, unflatten_connections
from .population import create_next_generation_then_speciate


class FunctionFactory:
    def __init__(self, config):
        self.config = config

        self.expand_coe = config.basic.expands_coe
        self.precompile_times = config.basic.pre_compile_times
        self.compiled_function = {}
        self.time_cost = {}

        self.load_config_vals(config)

        self.create_topological_sort_with_args()
        self.create_single_forward_with_args()
        self.create_update_speciate_with_args()

    def load_config_vals(self, config):
        self.compatibility_threshold = self.config.neat.species.compatibility_threshold

        self.problem_batch = config.basic.problem_batch

        self.pop_size = config.neat.population.pop_size

        self.disjoint_coe = config.neat.genome.compatibility_disjoint_coefficient
        self.compatibility_coe = config.neat.genome.compatibility_weight_coefficient

        self.num_inputs = config.basic.num_inputs
        self.num_outputs = config.basic.num_outputs
        self.input_idx = np.arange(self.num_inputs)
        self.output_idx = np.arange(self.num_inputs, self.num_inputs + self.num_outputs)

        bias = config.neat.gene.bias
        self.bias_mean = bias.init_mean
        self.bias_std = bias.init_stdev
        self.bias_mutate_strength = bias.mutate_power
        self.bias_mutate_rate = bias.mutate_rate
        self.bias_replace_rate = bias.replace_rate

        response = config.neat.gene.response
        self.response_mean = response.init_mean
        self.response_std = response.init_stdev
        self.response_mutate_strength = response.mutate_power
        self.response_mutate_rate = response.mutate_rate
        self.response_replace_rate = response.replace_rate

        weight = config.neat.gene.weight
        self.weight_mean = weight.init_mean
        self.weight_std = weight.init_stdev
        self.weight_mutate_strength = weight.mutate_power
        self.weight_mutate_rate = weight.mutate_rate
        self.weight_replace_rate = weight.replace_rate

        activation = config.neat.gene.activation
        self.act_default = act_name2key[activation.default]
        self.act_list = np.array([act_name2key[name] for name in activation.options])
        self.act_replace_rate = activation.mutate_rate

        aggregation = config.neat.gene.aggregation
        self.agg_default = agg_name2key[aggregation.default]
        self.agg_list = np.array([agg_name2key[name] for name in aggregation.options])
        self.agg_replace_rate = aggregation.mutate_rate

        enabled = config.neat.gene.enabled
        self.enabled_reverse_rate = enabled.mutate_rate

        genome = config.neat.genome
        self.add_node_rate = genome.node_add_prob
        self.delete_node_rate = genome.node_delete_prob
        self.add_connection_rate = genome.conn_add_prob
        self.delete_connection_rate = genome.conn_delete_prob
        self.single_structure_mutate = genome.single_structural_mutation

    def create_initialize(self, N, C):
        func = partial(
            initialize_genomes,
            pop_size=self.pop_size,
            N=N,
            C=C,
            num_inputs=self.num_inputs,
            num_outputs=self.num_outputs,
            default_bias=self.bias_mean,
            default_response=self.response_mean,
            default_act=self.act_default,
            default_agg=self.agg_default,
            default_weight=self.weight_mean
        )
        return func

    def create_update_speciate_with_args(self):
        species_kwargs = {
            "disjoint_coe": self.disjoint_coe,
            "compatibility_coe": self.compatibility_coe,
            "compatibility_threshold": self.compatibility_threshold
        }

        mutate_kwargs = {
            "input_idx": self.input_idx,
            "output_idx": self.output_idx,
            "bias_mean": self.bias_mean,
            "bias_std": self.bias_std,
            "bias_mutate_strength": self.bias_mutate_strength,
            "bias_mutate_rate": self.bias_mutate_rate,
            "bias_replace_rate": self.bias_replace_rate,
            "response_mean": self.response_mean,
            "response_std": self.response_std,
            "response_mutate_strength": self.response_mutate_strength,
            "response_mutate_rate": self.response_mutate_rate,
            "response_replace_rate": self.response_replace_rate,
            "weight_mean": self.weight_mean,
            "weight_std": self.weight_std,
            "weight_mutate_strength": self.weight_mutate_strength,
            "weight_mutate_rate": self.weight_mutate_rate,
            "weight_replace_rate": self.weight_replace_rate,
            "act_default": self.act_default,
            "act_list": self.act_list,
            "act_replace_rate": self.act_replace_rate,
            "agg_default": self.agg_default,
            "agg_list": self.agg_list,
            "agg_replace_rate": self.agg_replace_rate,
            "enabled_reverse_rate": self.enabled_reverse_rate,
            "add_node_rate": self.add_node_rate,
            "delete_node_rate": self.delete_node_rate,
            "add_connection_rate": self.add_connection_rate,
            "delete_connection_rate": self.delete_connection_rate,
        }

        self.update_speciate_with_args = partial(
            create_next_generation_then_speciate,
            species_kwargs=species_kwargs,
            mutate_kwargs=mutate_kwargs
        )

    def create_update_speciate(self, N, C, S):
        key = ("update_speciate", N, C, S)
        if key not in self.compiled_function:
            self.compile_update_speciate(N, C, S)
        return self.compiled_function[key]

    def compile_update_speciate(self, N, C, S):
        func = self.update_speciate_with_args
        randkey_lower = np.zeros((2,), dtype=np.uint32)
        pop_nodes_lower = np.zeros((self.pop_size, N, 5))
        pop_cons_lower = np.zeros((self.pop_size, C, 4))
        winner_part_lower = np.zeros((self.pop_size,), dtype=np.int32)
        loser_part_lower = np.zeros((self.pop_size,), dtype=np.int32)
        elite_mask_lower = np.zeros((self.pop_size,), dtype=bool)
        new_node_keys_start_lower = np.zeros((self.pop_size,), dtype=np.int32)
        pre_spe_center_nodes_lower = np.zeros((S, N, 5))
        pre_spe_center_cons_lower = np.zeros((S, C, 4))
        species_keys = np.zeros((S,), dtype=np.int32)
        new_species_keys_lower = 0
        compiled_func = jit(func).lower(
            randkey_lower,
            pop_nodes_lower,
            pop_cons_lower,
            winner_part_lower,
            loser_part_lower,
            elite_mask_lower,
            new_node_keys_start_lower,
            pre_spe_center_nodes_lower,
            pre_spe_center_cons_lower,
            species_keys,
            new_species_keys_lower,
        ).compile()
        self.compiled_function[("update_speciate", N, C, S)] = compiled_func

    def create_topological_sort_with_args(self):
        self.topological_sort_with_args = topological_sort

    def compile_topological_sort(self, n):
        func = self.topological_sort_with_args
        nodes_lower = np.zeros((n, 5))
        connections_lower = np.zeros((2, n, n))
        func = jit(func).lower(nodes_lower, connections_lower).compile()
        self.compiled_function[('topological_sort', n)] = func

    def create_topological_sort(self, n):
        key = ('topological_sort', n)
        if key not in self.compiled_function:
            self.compile_topological_sort(n)
        return self.compiled_function[key]

    def compile_topological_sort_batch(self, n):
        func = self.topological_sort_with_args
        func = vmap(func)
        nodes_lower = np.zeros((self.pop_size, n, 5))
        connections_lower = np.zeros((self.pop_size, 2, n, n))
        func = jit(func).lower(nodes_lower, connections_lower).compile()
        self.compiled_function[('topological_sort_batch', n)] = func

    def create_topological_sort_batch(self, n):
        key = ('topological_sort_batch', n)
        if key not in self.compiled_function:
            self.compile_topological_sort_batch(n)
        return self.compiled_function[key]

    def create_single_forward_with_args(self):
        func = partial(
            forward_single,
            input_idx=self.input_idx,
            output_idx=self.output_idx
        )
        self.single_forward_with_args = func

    def compile_single_forward(self, n):
        """
        single input for a genome
        :param n:
        :return:
        """
        func = self.single_forward_with_args
        inputs_lower = np.zeros((self.num_inputs,))
        cal_seqs_lower = np.zeros((n,), dtype=np.int32)
        nodes_lower = np.zeros((n, 5))
        connections_lower = np.zeros((2, n, n))
        func = jit(func).lower(inputs_lower, cal_seqs_lower, nodes_lower, connections_lower).compile()
        self.compiled_function[('single_forward', n)] = func

    def compile_pop_forward(self, n):
        func = self.single_forward_with_args
        func = vmap(func, in_axes=(None, 0, 0, 0))

        inputs_lower = np.zeros((self.num_inputs,))
        cal_seqs_lower = np.zeros((self.pop_size, n), dtype=np.int32)
        nodes_lower = np.zeros((self.pop_size, n, 5))
        connections_lower = np.zeros((self.pop_size, 2, n, n))
        func = jit(func).lower(inputs_lower, cal_seqs_lower, nodes_lower, connections_lower).compile()
        self.compiled_function[('pop_forward', n)] = func

    def compile_batch_forward(self, n):
        func = self.single_forward_with_args
        func = vmap(func, in_axes=(0, None, None, None))

        inputs_lower = np.zeros((self.problem_batch, self.num_inputs))
        cal_seqs_lower = np.zeros((n,), dtype=np.int32)
        nodes_lower = np.zeros((n, 5))
        connections_lower = np.zeros((2, n, n))
        func = jit(func).lower(inputs_lower, cal_seqs_lower, nodes_lower, connections_lower).compile()
        self.compiled_function[('batch_forward', n)] = func

    def create_batch_forward(self, n):
        key = ('batch_forward', n)
        if key not in self.compiled_function:
            self.compile_batch_forward(n)
        if self.debug:
            def debug_batch_forward(*args):
                return self.compiled_function[key](*args).block_until_ready()

            return debug_batch_forward
        else:
            return self.compiled_function[key]

    def compile_pop_batch_forward(self, n):
        func = self.single_forward_with_args
        func = vmap(func, in_axes=(0, None, None, None))  # batch_forward
        func = vmap(func, in_axes=(None, 0, 0, 0))  # pop_batch_forward

        inputs_lower = np.zeros((self.problem_batch, self.num_inputs))
        cal_seqs_lower = np.zeros((self.pop_size, n), dtype=np.int32)
        nodes_lower = np.zeros((self.pop_size, n, 5))
        connections_lower = np.zeros((self.pop_size, 2, n, n))

        func = jit(func).lower(inputs_lower, cal_seqs_lower, nodes_lower, connections_lower).compile()
        self.compiled_function[('pop_batch_forward', n)] = func

    def create_pop_batch_forward(self, n):
        key = ('pop_batch_forward', n)
        if key not in self.compiled_function:
            self.compile_pop_batch_forward(n)
        if self.debug:
            def debug_pop_batch_forward(*args):
                return self.compiled_function[key](*args).block_until_ready()

            return debug_pop_batch_forward
        else:
            return self.compiled_function[key]

    def ask_pop_batch_forward(self, pop_nodes, pop_cons):
        n, c = pop_nodes.shape[1], pop_cons.shape[1]
        batch_unflatten_func = self.create_batch_unflatten_connections(n, c)
        pop_cons = batch_unflatten_func(pop_nodes, pop_cons)
        ts = self.create_topological_sort_batch(n)
        pop_cal_seqs = ts(pop_nodes, pop_cons)

        forward_func = self.create_pop_batch_forward(n)

        def debug_forward(inputs):
            return forward_func(inputs, pop_cal_seqs, pop_nodes, pop_cons)

        return debug_forward

    def ask_batch_forward(self, nodes, connections):
        n = nodes.shape[0]
        ts = self.create_topological_sort(n)
        cal_seqs = ts(nodes, connections)
        forward_func = self.create_batch_forward(n)

        def debug_forward(inputs):
            return forward_func(inputs, cal_seqs, nodes, connections)

        return debug_forward

    def compile_batch_unflatten_connections(self, n, c):
        func = unflatten_connections
        func = vmap(func)
        pop_nodes_lower = np.zeros((self.pop_size, n, 5))
        pop_connections_lower = np.zeros((self.pop_size, c, 4))
        func = jit(func).lower(pop_nodes_lower, pop_connections_lower).compile()
        self.compiled_function[('batch_unflatten_connections', n, c)] = func

    def create_batch_unflatten_connections(self, n, c):
        key = ('batch_unflatten_connections', n, c)
        if key not in self.compiled_function:
            self.compile_batch_unflatten_connections(n, c)
        if self.debug:
            def debug_batch_unflatten_connections(*args):
                return self.compiled_function[key](*args).block_until_ready()

            return debug_batch_unflatten_connections
        else:
            return self.compiled_function[key]
