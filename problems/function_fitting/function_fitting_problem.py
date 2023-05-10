import numpy as np
import jax

from problems import Problem


class FunctionFittingProblem(Problem):
    def __init__(self, num_inputs, num_outputs, batch, inputs, target, loss='MSE'):
        self.forward_way = 'pop_batch'
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.batch = batch
        self.inputs = inputs
        self.target = target
        self.loss = loss
        super().__init__(self.forward_way, self.num_inputs, self.num_outputs, self.batch)

    def evaluate(self, batch_forward_func):
        out = batch_forward_func(self.inputs)
        out = jax.device_get(out)
        fitnesses = 1 - np.mean((self.target - out) ** 2, axis=(1, 2))
        return fitnesses.tolist()
