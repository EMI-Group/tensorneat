import numpy as np
from jax import jit, vmap

from .genome import create_forward, topological_sort, unflatten_connections
from .operations import create_next_generation_then_speciate

def hash_symbols(symbols):
    return symbols['P'], symbols['N'], symbols['C'], symbols['S']


class FunctionFactory:
    """
    Creates and compiles functions used in the NEAT pipeline.
    """

    def __init__(self, config, jit_config):
        self.config = config
        self.jit_config = jit_config

        self.func_dict = {}
        self.function_info = {}

        # (inputs_nums, ) -> (outputs_nums, )
        forward = create_forward(config)  # input size (inputs_nums, )

        # (batch_size, inputs_nums) -> (batch_size, outputs_nums)
        batch_forward = vmap(forward, in_axes=(0, None, None, None))

        # (pop_size, batch_size, inputs_nums) -> (pop_size, batch_size, outputs_nums)
        pop_batch_forward = vmap(batch_forward, in_axes=(0, 0, 0, 0))

        # (batch_size, inputs_nums) -> (pop_size, batch_size, outputs_nums)
        common_forward = vmap(batch_forward, in_axes=(None, 0, 0, 0))


        self.function_info = {
            "pop_unflatten_connections": {
                'func': vmap(unflatten_connections),
                'lowers': [
                    {'shape': ('P', 'N', 5), 'type': np.float32},
                    {'shape': ('P', 'C', 4), 'type': np.float32}
                ]
            },

            "pop_topological_sort": {
                'func': vmap(topological_sort),
                'lowers': [
                    {'shape': ('P', 'N', 5), 'type': np.float32},
                    {'shape': ('P', 2, 'N', 'N'), 'type': np.float32},
                ]
            },

            "batch_forward": {
                'func': batch_forward,
                'lowers': [
                    {'shape': (config['batch_size'], config['num_inputs']), 'type': np.float32},
                    {'shape': ('N', ), 'type': np.int32},
                    {'shape': ('N', 5), 'type': np.float32},
                    {'shape': (2, 'N', 'N'), 'type': np.float32}
                ]
            },

            "pop_batch_forward": {
                'func': pop_batch_forward,
                'lowers': [
                    {'shape': ('P', config['batch_size'], config['num_inputs']), 'type': np.float32},
                    {'shape': ('P', 'N'), 'type': np.int32},
                    {'shape': ('P', 'N', 5), 'type': np.float32},
                    {'shape': ('P', 2, 'N', 'N'), 'type': np.float32}
                ]
            },

            'common_forward': {
                'func': common_forward,
                'lowers': [
                    {'shape': (config['batch_size'], config['num_inputs']), 'type': np.float32},
                    {'shape': ('P', 'N'), 'type': np.int32},
                    {'shape': ('P', 'N', 5), 'type': np.float32},
                    {'shape': ('P', 2, 'N', 'N'), 'type': np.float32}
                ]
            },

            'create_next_generation_then_speciate': {
                'func': create_next_generation_then_speciate,
                'lowers': [
                    {'shape': (2, ), 'type': np.uint32},  # rand_key
                    {'shape': ('P', 'N', 5), 'type': np.float32},  # pop_nodes
                    {'shape': ('P', 'C', 4), 'type': np.float32},  # pop_cons
                    {'shape': ('P', ), 'type': np.int32},  # winner
                    {'shape': ('P', ), 'type': np.int32},  # loser
                    {'shape': ('P', ), 'type': bool},  # elite_mask
                    {'shape': ('P',), 'type': np.int32},  # new_node_keys
                    {'shape': ('S', 'N', 5), 'type': np.float32},  # center_nodes
                    {'shape': ('S', 'C', 4), 'type': np.float32},  # center_cons
                    {'shape': ('S', ), 'type': np.int32},  # species_keys
                    {'shape': (), 'type': np.int32},  # new_species_key_start
                    "jit_config"
                ]
            }
        }


    def get(self, name, symbols):
        if (name, hash_symbols(symbols)) not in self.func_dict:
            self.compile(name, symbols)
        return self.func_dict[name, hash_symbols(symbols)]

    def compile(self, name, symbols):
        # prepare function prototype
        func = self.function_info[name]['func']

        # prepare lower operands
        lowers_operands = []
        for lower in self.function_info[name]['lowers']:
            if isinstance(lower, dict):
                shape = list(lower['shape'])
                for i, s in enumerate(shape):
                    if s in symbols:
                        shape[i] = symbols[s]
                    assert isinstance(shape[i], int)
                lowers_operands.append(np.zeros(shape, dtype=lower['type']))

            elif lower == "jit_config":
                lowers_operands.append(self.jit_config)

            else:
                raise ValueError("Invalid lower operand")

        # compile
        compiled_func = jit(func).lower(*lowers_operands).compile()

        # save for reuse
        self.func_dict[name, hash_symbols(symbols)] = compiled_func
