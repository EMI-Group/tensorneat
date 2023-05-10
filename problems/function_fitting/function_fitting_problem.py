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

    def evaluate(self, pop_batch_forward):
        outs = pop_batch_forward(self.inputs)
        outs = jax.device_get(outs)
        fitnesses = -np.mean((self.target - outs) ** 2, axis=(1, 2))
        return fitnesses.tolist()

    def draw(self, batch_func):
        outs = batch_func(self.inputs)
        outs = jax.device_get(outs)
        print(outs)
        from matplotlib import pyplot as plt
        plt.xlabel('x')
        plt.ylabel('y')
        plt.plot(self.inputs, self.target, color='red', label='target')
        plt.plot(self.inputs, outs, color='blue', label='predict')
        plt.legend()
        plt.show()

    def print(self, batch_func):
        outs = batch_func(self.inputs)
        outs = jax.device_get(outs)
        print(outs)