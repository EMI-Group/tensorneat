import jax, jax.numpy as jnp
from utils import Act
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
            activation_options=(Act.tanh,),
            activation_default=Act.tanh,
        )
    )

    transformed = genome.transform(nodes, conns)
    seq, nodes, conns = transformed
    print(seq)

    exit(0)
    # print(*transformed, sep='\n')

    key = jax.random.key(0)
    dummy_input = jnp.zeros((8,))
    output = genome.forward(dummy_input, transformed)
    print(output)


if __name__ == '__main__':
    a = jnp.array([1, 3, 5, 6, 8])
    b = jnp.array([1, 2, 3])
    print(jnp.isin(a, b))
    # main()
