import numpy as np
from jax import jit, vmap

from .genome.forward import create_forward
from .genome.utils import unflatten_connections
from .genome.graph import topological_sort


def hash_symbols(symbols):
    return symbols['P'], symbols['N'], symbols['C'], symbols['S']


class FunctionFactory:
    """
    Creates and compiles functions used in the NEAT pipeline.
    """

    def __init__(self, config):
        self.config = config
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
            shape = list(lower['shape'])
            for i, s in enumerate(shape):
                if s in symbols:
                    shape[i] = symbols[s]
                assert isinstance(shape[i], int)
            lowers_operands.append(np.zeros(shape, dtype=lower['type']))

        # compile
        compiled_func = jit(func).lower(*lowers_operands).compile()

        # save for reuse
        self.func_dict[name, hash_symbols(symbols)] = compiled_func
