class Problem:
    def __init__(self, forward_way, num_inputs, num_outputs, batch):
        self.forward_way = forward_way
        self.batch = batch
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs

    def refactor_config(self, config):
        config.basic.forward_way = self.forward_way
        config.basic.num_inputs = self.num_inputs
        config.basic.num_outputs = self.num_outputs
        config.basic.problem_batch = self.batch

    def evaluate(self, batch_forward_func):
        pass
