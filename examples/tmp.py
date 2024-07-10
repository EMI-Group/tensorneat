import jax, jax.numpy as jnp

from tensorneat.algorithm import NEAT
from tensorneat.algorithm.neat import DefaultGenome

key = jax.random.key(0)
genome = DefaultGenome(num_inputs=5, num_outputs=3, max_nodes=100, max_conns=500, init_hidden_layers=())
state = genome.setup()
nodes, conns = genome.initialize(state, key)
print(genome.repr(state, nodes, conns))
