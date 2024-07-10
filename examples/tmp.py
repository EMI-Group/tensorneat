import jax, jax.numpy as jnp

from tensorneat.algorithm import NEAT
from tensorneat.algorithm.neat import DefaultGenome

key = jax.random.key(0)
genome = DefaultGenome(num_inputs=5, num_outputs=3, init_hidden_layers=(1, ))
state = genome.setup()
nodes, conns = genome.initialize(state, key)
print(genome.repr(state, nodes, conns))
