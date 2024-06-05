import jax.numpy as jnp

from pipeline import Pipeline
from algorithm.neat import *

from problem.rl_env.jumanji.jumanji_2048 import Jumanji_2048
from utils import Act, Agg

if __name__ == "__main__":
    pipeline = Pipeline(
        algorithm=NEAT(
            species=DefaultSpecies(
                genome=DefaultGenome(
                    num_inputs=16,
                    num_outputs=4,
                    max_nodes=100,
                    max_conns=1000,
                    node_gene=DefaultNodeGene(
                        activation_default=Act.sigmoid,
                        activation_options=(Act.sigmoid, Act.relu, Act.tanh, Act.identity, Act.inv),
                        aggregation_default=Agg.sum,
                        aggregation_options=(Agg.sum, Agg.mean, Agg.max, Agg.product),
                    ),
                    mutation=DefaultMutation(
                        node_add=0.03,
                        conn_add=0.03,
                    )
                ),
                pop_size=10000,
                species_size=100,
                survival_threshold=0.01,
            ),
        ),
        problem=Jumanji_2048(
            max_step=10000,
            repeat_times=5
        ),
        generation_limit=10000,
        fitness_target=13000,
    )

    # initialize state
    state = pipeline.setup()
    # print(state)
    # run until terminate
    state, best = pipeline.auto_run(state)
