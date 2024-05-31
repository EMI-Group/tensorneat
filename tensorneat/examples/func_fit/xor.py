from pipeline import Pipeline
from algorithm.neat import *

from problem.func_fit import XOR3d
from utils import Act

if __name__ == "__main__":
    pipeline = Pipeline(
        algorithm=NEAT(
            species=DefaultSpecies(
                genome=DefaultGenome(
                    num_inputs=3,
                    num_outputs=1,
                    max_nodes=100,
                    max_conns=200,
                    node_gene=DefaultNodeGene(
                        activation_default=Act.tanh,
                        activation_options=(Act.tanh,),
                    ),
                    output_transform=Act.sigmoid,  # the activation function for output node
                    mutation=DefaultMutation(
                        node_add=0.05,
                        conn_add=0.05,
                        node_delete=0.05,
                        conn_delete=0.05,
                    ),
                ),
                pop_size=1000,
                species_size=20,
                compatibility_threshold=2,
                survival_threshold=0.01,  # magic
            ),
        ),
        problem=XOR3d(),
        generation_limit=10000,
        fitness_target=-1e-8,
    )

    # initialize state
    state = pipeline.setup()
    # print(state)
    # run until terminate
    state, best = pipeline.auto_run(state)
    # show result
    pipeline.show(state, best)
