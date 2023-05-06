import numpy as np
from jax import numpy as jnp

from algorithms.neat.genome.genome import analysis
from algorithms.neat.genome import create_forward_function


error_nodes = np.load('error_nodes.npy')
error_connections = np.load('error_connections.npy')

node_dict, connection_dict = analysis(error_nodes, error_connections, np.array([0, 1]), np.array([2, ]))
print(node_dict, connection_dict, sep='\n')

N = error_nodes.shape[0]

xor_inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])

func = create_forward_function(error_nodes, error_connections, N, jnp.array([0, 1]), jnp.array([2, ]),
                               batch=True, debug=True)
out = func(np.array([1, 0]))

print(error_nodes)
print(error_connections)
print(out)