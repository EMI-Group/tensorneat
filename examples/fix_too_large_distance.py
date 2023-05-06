import numpy as np
from algorithms.neat.genome import distance

r_nodes = np.load('too_large_distance_r_nodes.npy')
r_connections = np.load('too_large_distance_r_connections.npy')
nodes = np.load('too_large_distance_nodes.npy')
connections = np.load('too_large_distance_connections.npy')

d1 = distance(r_nodes, r_connections, nodes, connections)
d2 = distance(nodes, connections, r_nodes, r_connections)
print(d1, d2)