"""
xor problem in multiple dimensions
"""

from itertools import product
import numpy as np


class EnhanceLogic:
    def __init__(self, name="xor", n=2):
        self.name = name
        self.n = n
        self.num_inputs = n
        self.num_outputs = 1
        self.batch = 2 ** n
        self.forward_way = 'pop_batch'

        self.inputs = np.array(generate_permutations(n), dtype=np.float32)

        if self.name == "xor":
            self.outputs = np.sum(self.inputs, axis=1) % 2
        elif self.name == "and":
            self.outputs = np.all(self.inputs==1, axis=1)
        elif self.name == "or":
            self.outputs = np.any(self.inputs==1, axis=1)
        else:
            raise NotImplementedError("Only support xor, and, or")
        self.outputs = self.outputs[:, np.newaxis]


    def refactor_config(self, config):
        config.basic.forward_way = self.forward_way
        config.basic.num_inputs = self.num_inputs
        config.basic.num_outputs = self.num_outputs
        config.basic.problem_batch = self.batch


    def ask_for_inputs(self):
        return self.inputs

    def evaluate_predict(self, predict):
        # print((predict - self.outputs) ** 2)
        return -np.mean((predict - self.outputs) ** 2)



def generate_permutations(n):
    permutations = [list(i) for i in product([0, 1], repeat=n)]

    return permutations


if __name__ == '__main__':
    _ = EnhanceLogic(4)
