import numpy as np

from .func_fit import FuncFit


class XOR(FuncFit):
    @property
    def inputs(self):
        return np.array(
            [[0, 0], [0, 1], [1, 0], [1, 1]],
            dtype=np.float32,
        )

    @property
    def targets(self):
        return np.array(
            [[0], [1], [1], [0]],
            dtype=np.float32,
        )

    @property
    def input_shape(self):
        return 4, 2

    @property
    def output_shape(self):
        return 4, 1
