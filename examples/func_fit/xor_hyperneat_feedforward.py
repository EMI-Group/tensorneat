import jax.numpy as jnp

from tensorneat.pipeline import Pipeline
from tensorneat.algorithm.neat import NEAT
from tensorneat.algorithm.hyperneat import HyperNEATFeedForward, MLPSubstrate
from tensorneat.genome import DefaultGenome
from tensorneat.common import ACT

from tensorneat.problem.func_fit import XOR3d

if __name__ == "__main__":
    pipeline = Pipeline(
        algorithm=HyperNEATFeedForward(
            substrate=MLPSubstrate(
                layers=[4, 5, 5, 5, 1], coor_range=(-5.0, 5.0, -5.0, 5.0)
            ),
            neat=NEAT(
                pop_size=10000,
                species_size=20,
                survival_threshold=0.01,
                genome=DefaultGenome(
                    num_inputs=4,  # size of query coors
                    num_outputs=1,
                    init_hidden_layers=(),
                    output_transform=ACT.tanh,
                ),
            ),
            activation=ACT.tanh,
            output_transform=ACT.sigmoid,
        ),
        problem=XOR3d(),
        generation_limit=1000,
        fitness_target=-1e-5,
    )

    # initialize state
    state = pipeline.setup()
    # print(state)
    # run until terminate
    state, best = pipeline.auto_run(state)
    # show result
    pipeline.show(state, best)

    # visualize cppn
    cppn_genome = pipeline.algorithm.neat.genome
    cppn_network = cppn_genome.network_dict(state, *best)
    cppn_genome.visualize(cppn_network, save_path="./imgs/cppn_network.svg")

    # visualize hyperneat genome
    hyperneat_genome = pipeline.algorithm.hyper_genome
    # use cppn to calculate the weights in hyperneat genome
    # return seqs, nodes, conns, u_conns
    _, hyperneat_nodes, hyperneat_conns, _ = pipeline.algorithm.transform(state, best)
    # mutate the connection with weight 0 (to visualize the network rather the substrate)
    hyperneat_conns = jnp.where(
        hyperneat_conns[:, 2][:, None] == 0, jnp.nan, hyperneat_conns
    )
    hyperneat_network = hyperneat_genome.network_dict(
        state, hyperneat_nodes, hyperneat_conns
    )
    hyperneat_genome.visualize(
        hyperneat_network, save_path="./imgs/hyperneat_network.svg"
    )
