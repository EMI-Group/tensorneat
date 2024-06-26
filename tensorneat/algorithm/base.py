from utils import State, StatefulBaseClass


class BaseAlgorithm(StatefulBaseClass):
    def ask(self, state: State):
        """require the population to be evaluated"""
        raise NotImplementedError

    def tell(self, state: State, fitness):
        """update the state of the algorithm"""
        raise NotImplementedError

    def transform(self, state, individual):
        """transform the genome into a neural network"""
        raise NotImplementedError

    def restore(self, state, transformed):
        raise NotImplementedError

    def forward(self, state, transformed, inputs):
        raise NotImplementedError

    def update_by_batch(self, state, batch_input, transformed):
        raise NotImplementedError

    @property
    def num_inputs(self):
        raise NotImplementedError

    @property
    def num_outputs(self):
        raise NotImplementedError

    @property
    def pop_size(self):
        raise NotImplementedError

    def member_count(self, state: State):
        # to analysis the species
        raise NotImplementedError

    def generation(self, state: State):
        # to analysis the algorithm
        raise NotImplementedError
