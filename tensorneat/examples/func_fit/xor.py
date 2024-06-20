from pipeline import Pipeline
from algorithm.neat import *

from problem.func_fit import XOR3d
from utils import ACT_ALL, AGG_ALL, Act, Agg

if __name__ == "__main__":
    pipeline = Pipeline(
        algorithm=NEAT(
            species=DefaultSpecies(
                genome=DenseInitialize(
                    num_inputs=3,
                    num_outputs=1,
                    max_nodes=50,
                    max_conns=100,
                    node_gene=DefaultNodeGene(
                        activation_default=Act.tanh,
                        # activation_options=(Act.tanh,),
                        activation_options=ACT_ALL,
                        aggregation_default=Agg.sum,
                        # aggregation_options=(Agg.sum,),
                        aggregation_options=AGG_ALL,
                    ),
                    output_transform=Act.standard_sigmoid,  # the activation function for output node
                    mutation=DefaultMutation(
                        node_add=0.1,
                        conn_add=0.1,
                        node_delete=0,
                        conn_delete=0,
                    ),
                ),
                pop_size=10000,
                species_size=20,
                compatibility_threshold=2,
                survival_threshold=0.01,  # magic
            ),
        ),
        problem=XOR3d(),
        generation_limit=10000,
        fitness_target=-1e-3,
    )

    # initialize state
    state = pipeline.setup()
    # print(state)
    # run until terminate
    state, best = pipeline.auto_run(state)
    # show result
    pipeline.show(state, best)
    pipeline.save(state=state)
