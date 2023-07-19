import jax.numpy as jnp

from algorithm.neat.genome.graph import topological_sort, check_cycles
from algorithm.neat.utils import I_INT

nodes = jnp.array([
    [0],
    [1],
    [2],
    [3],
    [jnp.nan]
])

# {(0, 2), (1, 2), (1, 3), (2, 3)}
conns = jnp.array([
    [0, 0, 1, 0, 0],
    [0, 0, 1, 1, 0],
    [0, 0, 0, 1, 0],
    [0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0]
])


def test_topological_sort():
    assert jnp.all(topological_sort(nodes, conns) == jnp.array([0, 1, 2, 3, I_INT]))


def test_check_cycles():
    assert check_cycles(nodes, conns, 3, 2)
    assert ~check_cycles(nodes, conns, 2, 3)
    assert ~check_cycles(nodes, conns, 0, 3)
    assert ~check_cycles(nodes, conns, 1, 0)
