import numpy as np

from . import FunctionFittingProblem


class Sin(FunctionFittingProblem):
    def __init__(self, size=100):
        self.num_inputs = 1
        self.num_outputs = 1
        self.batch = size
        self.inputs = np.linspace(0, np.pi, self.batch)[:, None]
        self.target = np.sin(self.inputs)
        print(self.inputs, self.target)
        super().__init__(self.num_inputs, self.num_outputs, self.batch, self.inputs, self.target)
