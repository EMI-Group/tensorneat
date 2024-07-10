import jax

from pipeline import Pipeline
from algorithm.neat import *
from algorithm.hyperneat import *
from tensorneat.common import Act

from problem.rl_env import GymNaxEnv

if __name__ == "__main__":
    pipeline = Pipeline(
        algorithm=HyperNEAT(
            substrate=FullSubstrate(
                input_coors=[
                    (-1, -1),
                    (-0.5, -1),
                    (0, -1),
                    (0.5, -1),
                    (1, -1),
                ],  # 4(problem inputs) + 1(bias)
                hidden_coors=[
                    (-1, -0.5),
                    (0.333, -0.5),
                    (-0.333, -0.5),
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
                    (-1, 1),
                    (1, 1),  # one output
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
                    pop_size=10000,
                    species_size=10,
                    compatibility_threshold=3.5,
                    survival_threshold=0.03,
                ),
            ),
            activation=Act.tanh,  # the activation function for output node in HyperNEAT
            activate_time=10,
            output_transform=jax.numpy.argmax,  # action of cartpole is in {0, 1}
        ),
        problem=GymNaxEnv(
            env_name="CartPole-v1",
        ),
        generation_limit=300,
        fitness_target=500,
    )

    # initialize state
    state = pipeline.setup()
    # print(state)
    # run until terminate
    state, best = pipeline.auto_run(state)
