import numpy as np

from . import FunctionFittingProblem


class DIY(FunctionFittingProblem):
    def __init__(self, func, size=100):
        self.num_inputs = 1
        self.num_outputs = 1
        self.batch = size
        self.inputs = np.linspace(0, 1, self.batch)[:, None]
        self.target = func(self.inputs)
        print(self.inputs, self.target)
        super().__init__(self.num_inputs, self.num_outputs, self.batch, self.inputs, self.target)
