import jax, jax.numpy as jnp

from tensorneat.algorithm import NEAT
from tensorneat.algorithm.neat import DefaultGenome, RecurrentGenome

key = jax.random.key(0)
genome = DefaultGenome(num_inputs=5, num_outputs=3, max_nodes=100, max_conns=500, init_hidden_layers=(1, 2 ,3))
state = genome.setup()
nodes, conns = genome.initialize(state, key)
print(genome.repr(state, nodes, conns))

inputs = jnp.array([1, 2, 3, 4, 5])
transformed = genome.transform(state, nodes, conns)
outputs = genome.forward(state, transformed, inputs)

print(outputs)

network = genome.network_dict(state, nodes, conns)
print(network)

genome.visualize(network)
