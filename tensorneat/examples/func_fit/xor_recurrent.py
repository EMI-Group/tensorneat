from pipeline import Pipeline
from algorithm.neat import *

from problem.func_fit import XOR3d
from utils.activation import ACT_ALL, Act

if __name__ == '__main__':
    pipeline = Pipeline(
        seed=0,
        algorithm=NEAT(
            species=DefaultSpecies(
                genome=RecurrentGenome(
                    num_inputs=3,
                    num_outputs=1,
                    max_nodes=50,
                    max_conns=100,
                    activate_time=5,
                    node_gene=DefaultNodeGene(
                        activation_options=ACT_ALL,
                        activation_replace_rate=0.2
                    ),
                    output_transform=Act.sigmoid
                ),
                pop_size=10000,
                species_size=10,
                compatibility_threshold=3.5,
                survival_threshold=0.03,
            ),
            mutation=DefaultMutation(
                node_add=0.05,
                conn_add=0.2,
                node_delete=0,
                conn_delete=0,
            )
        ),
        problem=XOR3d(),
        generation_limit=10000,
        fitness_target=-1e-8
    )

    # initialize state
    state = pipeline.setup()
    # print(state)
    # run until terminate
    state, best = pipeline.auto_run(state)
    # show result
    pipeline.show(state, best)
