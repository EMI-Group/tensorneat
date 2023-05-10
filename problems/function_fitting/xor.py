import numpy as np

from . import FunctionFittingProblem


class Xor(FunctionFittingProblem):
    def __init__(self):
        self.num_inputs = 2
        self.num_outputs = 1
        self.batch = 4
        self.inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float32)
        self.target = np.array([[0], [1], [1], [0]], dtype=np.float32)
        super().__init__(self.num_inputs, self.num_outputs, self.batch, self.inputs, self.target)
