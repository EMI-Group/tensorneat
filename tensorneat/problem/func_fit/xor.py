import numpy as np

from .func_fit import FuncFit


class XOR(FuncFit):

    def __init__(self, error_method: str = 'mse'):
        super().__init__(error_method)

    @property
    def inputs(self):
        return np.array([
            [0, 0],
            [0, 1],
            [1, 0],
            [1, 1]
        ])

    @property
    def targets(self):
        return np.array([
            [0],
            [1],
            [1],
            [0]
        ])

    @property
    def input_shape(self):
        return 4, 2

    @property
    def output_shape(self):
        return 4, 1
