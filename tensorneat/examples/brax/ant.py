from pipeline import Pipeline
from algorithm.neat import *

from problem.rl_env import BraxEnv
from utils import Act

if __name__ == "__main__":
    pipeline = Pipeline(
        algorithm=NEAT(
            species=DefaultSpecies(
                genome=DefaultGenome(
                    num_inputs=27,
                    num_outputs=8,
                    max_nodes=50,
                    max_conns=100,
                    node_gene=DefaultNodeGene(
                        activation_options=(Act.tanh,),
                        activation_default=Act.tanh,
                    ),
                    output_transform=Act.tanh,
                ),
                pop_size=1000,
                species_size=10,
            ),
        ),
        problem=BraxEnv(
            env_name="ant",
        ),
        generation_limit=10000,
        fitness_target=5000,
    )

    # initialize state
    state = pipeline.setup()
    # print(state)
    # run until terminate
    state, best = pipeline.auto_run(state)
