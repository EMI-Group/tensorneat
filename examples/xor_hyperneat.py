import jax
import numpy as np

from config import Config, BasicConfig, NeatConfig
from pipeline import Pipeline
from algorithm import NEAT, HyperNEAT
from algorithm.neat.gene import RecurrentGene, RecurrentGeneConfig
from algorithm.hyperneat.substrate import NormalSubstrate, NormalSubstrateConfig

xor_inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float32)
xor_outputs = np.array([[0], [1], [1], [0]], dtype=np.float32)


def evaluate(forward_func):
    """
    :param forward_func: (4: batch, 2: input size) -> (pop_size, 4: batch, 1: output size)
    :return:
    """
    outs = forward_func(xor_inputs)
    outs = jax.device_get(outs)
    fitnesses = 4 - np.sum((outs - xor_outputs) ** 2, axis=(1, 2))
    return fitnesses


if __name__ == '__main__':
    config = Config(
        basic=BasicConfig(
            fitness_target=3.99999,
            pop_size=10000
        ),
        neat=NeatConfig(
            network_type="recurrent",
            maximum_nodes=50,
            maximum_conns=100,
            inputs=4,
            outputs=1

        ),
        gene=RecurrentGeneConfig(
            activation_default="tanh",
            activation_options=("tanh",),
        ),
        substrate=NormalSubstrateConfig(),
    )
    neat = NEAT(config, RecurrentGene)
    hyperNEAT = HyperNEAT(config, neat, NormalSubstrate)

    pipeline = Pipeline(config, hyperNEAT)
    pipeline.auto_run(evaluate)
