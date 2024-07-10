from pipeline import Pipeline
from algorithm.neat import *
from algorithm.hyperneat import *
from tensorneat.common import Act

from problem.func_fit import XOR3d

if __name__ == "__main__":
    pipeline = Pipeline(
        algorithm=HyperNEAT(
            substrate=FullSubstrate(
                input_coors=[(-1, -1), (0.333, -1), (-0.333, -1), (1, -1)],  # 3(XOR3d inputs) + 1(bias)
                hidden_coors=[
                    (-1, -0.5), (0.333, -0.5), (-0.333, -0.5),
                    (1, -0.5),
                    (-1, 0),
                    (0.333, 0),
                    (-0.333, 0),
                    (1, 0),
                    (-1, 0.5),
                    (0.333, 0.5),
                    (-0.333, 0.5),
                    (1, 0.5),
                ],
                output_coors=[
                    (0, 1),  # one output
                ],
            ),
            neat=NEAT(
                species=DefaultSpecies(
                    genome=DefaultGenome(
                        num_inputs=4,  # [*coor1, *coor2]
                        num_outputs=1,  # the weight of connection between two coor1 and coor2
                        max_nodes=50,
                        max_conns=100,
                        node_gene=DefaultNodeGene(
                            activation_default=Act.tanh,
                            activation_options=(Act.tanh,),
                        ),
                        output_transform=Act.tanh,  # the activation function for output node in NEAT
                    ),
                    pop_size=1000,
                    species_size=10,
                    compatibility_threshold=2,
                    survival_threshold=0.03,
                ),
            ),
            activation=Act.tanh,
            activate_time=10,
            output_transform=Act.sigmoid,  # the activation function for output node in HyperNEAT
        ),
        problem=XOR3d(),
        generation_limit=300,
        fitness_target=-1e-6,
    )

    # initialize state
    state = pipeline.setup()
    # print(state)
    # run until terminate
    state, best = pipeline.auto_run(state)
    # show result
    pipeline.show(state, best)
