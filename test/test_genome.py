import jax
from algorithm.neat import *

genome = DefaultGenome(
    num_inputs=3,
    num_outputs=1,
    max_nodes=5,
    max_conns=10,
)


def test_output_work():
    randkey = jax.random.PRNGKey(0)
    state = genome.setup()
    nodes, conns = genome.initialize(state, randkey)
    transformed = genome.transform(state, nodes, conns)
    inputs = jax.random.normal(randkey, (3,))
    output = genome.forward(state, transformed, inputs)
    print(output)

    batch_inputs = jax.random.normal(randkey, (10, 3))
    batch_output = jax.vmap(genome.forward, in_axes=(None, None, 0))(
        state, transformed, batch_inputs
    )
    print(batch_output)

    assert True
