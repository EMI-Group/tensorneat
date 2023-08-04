import numpy as np

from .func_fit import FuncFit, FuncFitConfig


class XOR(FuncFit):

    def __init__(self, config: FuncFitConfig = FuncFitConfig()):
        self.config = config
        super().__init__(config)

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
        return (4, 2)

    @property
    def output_shape(self):
        return (4, 1)
