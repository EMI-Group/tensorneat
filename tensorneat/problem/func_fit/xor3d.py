import numpy as np

from .func_fit import FuncFit


class XOR3d(FuncFit):
    @property
    def inputs(self):
        return np.array(
            [
                [0, 0, 0],
                [0, 0, 1],
                [0, 1, 0],
                [0, 1, 1],
                [1, 0, 0],
                [1, 0, 1],
                [1, 1, 0],
                [1, 1, 1],
            ]
        )

    @property
    def targets(self):
        return np.array([[0], [1], [1], [0], [1], [0], [0], [1]])

    @property
    def input_shape(self):
        return 8, 3

    @property
    def output_shape(self):
        return 8, 1
