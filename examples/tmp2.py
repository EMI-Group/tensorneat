import jax, jax.numpy as jnp

from tensorneat.pipeline import Pipeline
from tensorneat.algorithm.neat import NEAT
from tensorneat.genome import DefaultGenome, DefaultNode, DefaultMutation, BiasNode
from tensorneat.problem.func_fit import CustomFuncFit
from tensorneat.common import Act, Agg


def pagie_polynomial(inputs):
    x, y = inputs
    return x + y


if __name__ == "__main__":
    genome=DefaultGenome(
        num_inputs=2,
        num_outputs=1,
        max_nodes=3,
        max_conns=2,
        init_hidden_layers=(),
        node_gene=BiasNode(
            activation_options=[Act.identity],
            aggregation_options=[Agg.sum],
        ),
        output_transform=Act.identity,
        mutation=DefaultMutation(
            node_add=0,
            node_delete=0,
            conn_add=0.0,
            conn_delete=0.0,
        )
    )
    randkey = jax.random.PRNGKey(42)
    state = genome.setup()
    nodes, conns = genome.initialize(state, randkey)
    print(genome)


