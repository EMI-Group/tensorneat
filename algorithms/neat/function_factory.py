"""
Lowers, compiles, and creates functions used in the NEAT pipeline.
"""
from functools import partial

import jax.random
import numpy as np
from jax import jit, vmap

from .genome import act_name2key, agg_name2key, initialize_genomes, mutate, distance, crossover
from .genome import topological_sort, forward_single


class FunctionFactory:
    def __init__(self, config, debug=False):
        self.config = config
        self.debug = debug

        self.init_N = config.basic.init_maximum_nodes
        self.expand_coe = config.basic.expands_coe
        self.precompile_times = config.basic.pre_compile_times
        self.compiled_function = {}

        self.load_config_vals(config)
        self.precompile()
        pass

    def load_config_vals(self, config):
        self.problem_batch = config.basic.problem_batch

        self.pop_size = config.neat.population.pop_size
        self.init_N = config.basic.init_maximum_nodes

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

    def create_initialize(self):
        func = partial(
            initialize_genomes,
            pop_size=self.pop_size,
            N=self.init_N,
            num_inputs=self.num_inputs,
            num_outputs=self.num_outputs,
            default_bias=self.bias_mean,
            default_response=self.response_mean,
            default_act=self.act_default,
            default_agg=self.agg_default,
            default_weight=self.weight_mean
        )
        if self.debug:
            def debug_initialize(*args):
                return func(*args)

            return debug_initialize
        else:
            return func

    def precompile(self):
        self.create_mutate_with_args()
        self.create_distance_with_args()
        self.create_crossover_with_args()
        self.create_topological_sort_with_args()
        self.create_single_forward_with_args()

        n = self.init_N
        print("start precompile")
        for _ in range(self.precompile_times):
            self.compile_mutate(n)
            self.compile_distance(n)
            self.compile_crossover(n)
            self.compile_topological_sort(n)
            self.compile_pop_batch_forward(n)
            n = int(self.expand_coe * n)

        # precompile other functions used in jax
        key = jax.random.PRNGKey(0)
        _ = jax.random.split(key, 3)
        _ = jax.random.split(key, self.pop_size * 2)
        _ = jax.random.split(key, self.pop_size)

        print("end precompile")

    def create_mutate_with_args(self):
        func = partial(
            mutate,
            input_idx=self.input_idx,
            output_idx=self.output_idx,
            bias_mean=self.bias_mean,
            bias_std=self.bias_std,
            bias_mutate_strength=self.bias_mutate_strength,
            bias_mutate_rate=self.bias_mutate_rate,
            bias_replace_rate=self.bias_replace_rate,
            response_mean=self.response_mean,
            response_std=self.response_std,
            response_mutate_strength=self.response_mutate_strength,
            response_mutate_rate=self.response_mutate_rate,
            response_replace_rate=self.response_replace_rate,
            weight_mean=self.weight_mean,
            weight_std=self.weight_std,
            weight_mutate_strength=self.weight_mutate_strength,
            weight_mutate_rate=self.weight_mutate_rate,
            weight_replace_rate=self.weight_replace_rate,
            act_default=self.act_default,
            act_list=self.act_list,
            act_replace_rate=self.act_replace_rate,
            agg_default=self.agg_default,
            agg_list=self.agg_list,
            agg_replace_rate=self.agg_replace_rate,
            enabled_reverse_rate=self.enabled_reverse_rate,
            add_node_rate=self.add_node_rate,
            delete_node_rate=self.delete_node_rate,
            add_connection_rate=self.add_connection_rate,
            delete_connection_rate=self.delete_connection_rate,
            single_structure_mutate=self.single_structure_mutate
        )
        self.mutate_with_args = func

    def compile_mutate(self, n):
        func = self.mutate_with_args
        rand_key_lower = np.zeros((self.pop_size, 2), dtype=np.uint32)
        nodes_lower = np.zeros((self.pop_size, n, 5))
        connections_lower = np.zeros((self.pop_size, 2, n, n))
        new_node_key_lower = np.zeros((self.pop_size,), dtype=np.int32)
        batched_mutate_func = jit(vmap(func)).lower(rand_key_lower, nodes_lower,
                                                    connections_lower, new_node_key_lower).compile()
        self.compiled_function[('mutate', n)] = batched_mutate_func

    def create_mutate(self, n):
        key = ('mutate', n)
        if key not in self.compiled_function:
            self.compile_mutate(n)
        if self.debug:
            def debug_mutate(*args):
                res_nodes, res_connections = self.compiled_function[key](*args)
                return res_nodes.block_until_ready(), res_connections.block_until_ready()

            return debug_mutate
        else:
            return self.compiled_function[key]

    def create_distance_with_args(self):
        func = partial(
            distance,
            disjoint_coe=self.disjoint_coe,
            compatibility_coe=self.compatibility_coe
        )
        self.distance_with_args = func

    def compile_distance(self, n):
        func = self.distance_with_args
        o2o_nodes1_lower = np.zeros((n, 5))
        o2o_connections1_lower = np.zeros((2, n, n))
        o2o_nodes2_lower = np.zeros((n, 5))
        o2o_connections2_lower = np.zeros((2, n, n))
        o2o_distance = jit(func).lower(o2o_nodes1_lower, o2o_connections1_lower,
                                       o2o_nodes2_lower, o2o_connections2_lower).compile()

        o2m_nodes2_lower = np.zeros((self.pop_size, n, 5))
        o2m_connections2_lower = np.zeros((self.pop_size, 2, n, n))
        o2m_distance = jit(vmap(func, in_axes=(None, None, 0, 0))).lower(o2o_nodes1_lower, o2o_connections1_lower,
                                                                         o2m_nodes2_lower,
                                                                         o2m_connections2_lower).compile()

        self.compiled_function[('o2o_distance', n)] = o2o_distance
        self.compiled_function[('o2m_distance', n)] = o2m_distance

    def create_distance(self, n):
        key1, key2 = ('o2o_distance', n), ('o2m_distance', n)
        if key1 not in self.compiled_function:
            self.compile_distance(n)
        if self.debug:

            def debug_o2o_distance(*args):
                return self.compiled_function[key1](*args).block_until_ready()

            def debug_o2m_distance(*args):
                return self.compiled_function[key2](*args).block_until_ready()

            return debug_o2o_distance, debug_o2m_distance
        else:
            return self.compiled_function[key1], self.compiled_function[key2]

    def create_crossover_with_args(self):
        self.crossover_with_args = crossover

    def compile_crossover(self, n):
        func = self.crossover_with_args
        randkey_lower = np.zeros((self.pop_size, 2), dtype=np.uint32)
        nodes1_lower = np.zeros((self.pop_size, n, 5))
        connections1_lower = np.zeros((self.pop_size, 2, n, n))
        nodes2_lower = np.zeros((self.pop_size, n, 5))
        connections2_lower = np.zeros((self.pop_size, 2, n, n))
        func = jit(vmap(func)).lower(randkey_lower, nodes1_lower, connections1_lower,
                                     nodes2_lower, connections2_lower).compile()
        self.compiled_function[('crossover', n)] = func

    def create_crossover(self, n):
        key = ('crossover', n)
        if key not in self.compiled_function:
            self.compile_crossover(n)
        if self.debug:

            def debug_crossover(*args):
                res_nodes, res_connections = self.compiled_function[key](*args)
                return res_nodes.block_until_ready(), res_connections.block_until_ready()

            return debug_crossover
        else:
            return self.compiled_function[key]

    def create_topological_sort_with_args(self):
        self.topological_sort_with_args = topological_sort

    def compile_topological_sort(self, n):
        func = self.topological_sort_with_args
        func = vmap(func)
        nodes_lower = np.zeros((self.pop_size, n, 5))
        connections_lower = np.zeros((self.pop_size, 2, n, n))
        func = jit(func).lower(nodes_lower, connections_lower).compile()
        self.compiled_function[('topological_sort', n)] = func

    def create_topological_sort(self, n):
        key = ('topological_sort', n)
        if key not in self.compiled_function:
            self.compile_topological_sort(n)
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

    def ask(self, pop_nodes, pop_connections):
        n = pop_nodes.shape[1]
        ts = self.create_topological_sort(n)
        pop_cal_seqs = ts(pop_nodes, pop_connections)

        forward_func = self.create_pop_batch_forward(n)

        def debug_forward(inputs):
            return forward_func(inputs, pop_cal_seqs, pop_nodes, pop_connections)

        return debug_forward

        # return partial(
        #     forward_func,
        #     cal_seqs=pop_cal_seqs,
        #     nodes=pop_nodes,
        #     connections=pop_connections
        # )
