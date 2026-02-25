from tensorneat.common import State, StatefulBaseClass, PolicyAPI, _wrap_stateless


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

    def get_forward(self):
        raise NotImplementedError

    def show_details(self, state: State, fitness):
        """Visualize the running details of the algorithm"""
        raise NotImplementedError
    
    def stateful_policy_api(self) -> PolicyAPI:
        """Give Stateful forward function of this algorithm"""
        return _wrap_stateless(self.get_forward())

    @property
    def num_inputs(self):
        raise NotImplementedError

    @property
    def num_outputs(self):
        raise NotImplementedError
