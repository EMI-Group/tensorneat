import jax, jax.numpy as jnp
from tensorneat.common import ACT
from algorithm.neat import *
import numpy as np


def main():
    node_path = "../examples/brax/nan_node.npy"
    conn_path = "../examples/brax/nan_conn.npy"
    nodes = np.load(node_path)
    conns = np.load(conn_path)
    nodes, conns = jax.device_put([nodes, conns])

    genome = DefaultGenome(
        num_inputs=8,
        num_outputs=2,
        max_nodes=20,
        max_conns=20,
        node_gene=DefaultNodeGene(
            activation_options=(ACT.tanh,),
            activation_default=ACT.tanh,
        ),
    )

    transformed = genome.transform(nodes, conns)
    print(*transformed, sep="\n")

    key = jax.random.key(0)
    dummy_input = jnp.zeros((8,))
    output = genome.forward(dummy_input, transformed)
    print(output)


if __name__ == "__main__":
    main()
