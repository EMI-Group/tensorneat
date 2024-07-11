import jax.numpy as jnp

from evox import Algorithm as EvoXAlgorithm, State as EvoXState, jit_class

from tensorneat.algorithm import BaseAlgorithm as TensorNEATAlgorithm
from tensorneat.common import State as TensorNEATState


@jit_class
class EvoXAlgorithmAdaptor(EvoXAlgorithm):
    def __init__(self, algorithm: TensorNEATAlgorithm):
        self.algorithm = algorithm
        self.fixed_state = None

    def setup(self, key):
        neat_algorithm_state = TensorNEATState(randkey=key)
        neat_algorithm_state = self.algorithm.setup(neat_algorithm_state)
        self.fixed_state = neat_algorithm_state
        return EvoXState(alg_state=neat_algorithm_state)

    def ask(self, state: EvoXState):
        population = self.algorithm.ask(state.alg_state)
        return population, state

    def tell(self, state: EvoXState, fitness):
        fitness = jnp.where(jnp.isnan(fitness), -jnp.inf, fitness)
        neat_algorithm_state = self.algorithm.tell(state.alg_state, fitness)
        return state.replace(alg_state=neat_algorithm_state)

    def transform(self, individual):
        return self.algorithm.transform(self.fixed_state, individual)

    def forward(self, transformed, inputs):
        return self.algorithm.forward(self.fixed_state, transformed, inputs)
