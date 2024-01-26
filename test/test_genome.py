from algorithm.neat import *
from utils import Act, Agg

import jax, jax.numpy as jnp


def test_default():

    # index, bias, response, activation, aggregation
    nodes = jnp.array([
        [0, 0, 1, 0, 0], # in[0]
        [1, 0, 1, 0, 0], # in[1]
        [2, 0.5, 1, 0, 0],  # out[0],
        [3, 1, 1, 0, 0],  # hidden[0],
        [4, -1, 1, 0, 0],  # hidden[1],
    ])

    # in_node, out_node, enable, weight
    conns = jnp.array([
        [0, 3, 1, 0.5], # in[0] -> hidden[0]
        [1, 4, 1, 0.5], # in[1] -> hidden[1]
        [3, 2, 1, 0.5], # hidden[0] -> out[0]
        [4, 2, 1, 0.5], # hidden[1] -> out[0]
    ])

    genome = DefaultGenome(
        num_inputs=2,
        num_outputs=1,
        node_gene=DefaultNodeGene(
            activation_default=Act.identity,
            activation_options=(Act.identity, ),
            aggregation_default=Agg.sum,
            aggregation_options=(Agg.sum, ),
        ),
    )

    transformed = genome.transform(nodes, conns)
    print(*transformed, sep='\n')

    inputs = jnp.array([[0, 0],[0, 1], [1, 0], [1, 1]])
    outputs = jax.jit(jax.vmap(genome.forward, in_axes=(0, None)))(inputs, transformed)
    print(outputs)
    assert jnp.allclose(outputs, jnp.array([[0.5], [0.75], [0.75], [1]]))
    # expected: [[0.5], [0.75], [0.75], [1]]

    print('\n-------------------------------------------------------\n')

    conns = conns.at[0, 2].set(False)  # disable in[0] -> hidden[0]
    print(conns)

    transformed = genome.transform(nodes, conns)
    print(*transformed, sep='\n')

    inputs = jnp.array([[0, 0],[0, 1], [1, 0], [1, 1]])
    outputs = jax.vmap(genome.forward, in_axes=(0, None))(inputs, transformed)
    print(outputs)
    assert jnp.allclose(outputs, jnp.array([[0], [0.25], [0], [0.25]]))
    # expected: [[0.5], [0.75], [0.5], [0.75]]


def test_recurrent():

    # index, bias, response, activation, aggregation
    nodes = jnp.array([
        [0, 0, 1, 0, 0], # in[0]
        [1, 0, 1, 0, 0], # in[1]
        [2, 0.5, 1, 0, 0],  # out[0],
        [3, 1, 1, 0, 0],  # hidden[0],
        [4, -1, 1, 0, 0],  # hidden[1],
    ])

    # in_node, out_node, enable, weight
    conns = jnp.array([
        [0, 3, 1, 0.5], # in[0] -> hidden[0]
        [1, 4, 1, 0.5], # in[1] -> hidden[1]
        [3, 2, 1, 0.5], # hidden[0] -> out[0]
        [4, 2, 1, 0.5], # hidden[1] -> out[0]
    ])

    genome = RecurrentGenome(
        num_inputs=2,
        num_outputs=1,
        node_gene=DefaultNodeGene(
            activation_default=Act.identity,
            activation_options=(Act.identity, ),
            aggregation_default=Agg.sum,
            aggregation_options=(Agg.sum, ),
        ),
        activate_time=3,
    )

    transformed = genome.transform(nodes, conns)
    print(*transformed, sep='\n')

    inputs = jnp.array([[0, 0],[0, 1], [1, 0], [1, 1]])
    outputs = jax.jit(jax.vmap(genome.forward, in_axes=(0, None)))(inputs, transformed)
    print(outputs)
    assert jnp.allclose(outputs, jnp.array([[0.5], [0.75], [0.75], [1]]))
    # expected: [[0.5], [0.75], [0.75], [1]]

    print('\n-------------------------------------------------------\n')

    conns = conns.at[0, 2].set(False)  # disable in[0] -> hidden[0]
    print(conns)

    transformed = genome.transform(nodes, conns)
    print(*transformed, sep='\n')

    inputs = jnp.array([[0, 0],[0, 1], [1, 0], [1, 1]])
    outputs = jax.vmap(genome.forward, in_axes=(0, None))(inputs, transformed)
    print(outputs)
    assert jnp.allclose(outputs, jnp.array([[0], [0.25], [0], [0.25]]))
    # expected: [[0.5], [0.75], [0.5], [0.75]]