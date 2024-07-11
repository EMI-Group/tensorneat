from tensorneat.pipeline import Pipeline
from tensorneat.algorithm.neat import NEAT
from tensorneat.genome import DefaultGenome, DefaultNodeGene, DefaultMutation
from tensorneat.problem.func_fit import XOR3d
from tensorneat.common import Act, Agg

if __name__ == "__main__":
    pipeline = Pipeline(
        algorithm=NEAT(
            pop_size=10000,
            species_size=20,
            compatibility_threshold=2,
            survival_threshold=0.01,
            genome=DefaultGenome(
                num_inputs=3,
                num_outputs=1,
                init_hidden_layers=(),
                node_gene=DefaultNodeGene(
                    activation_default=Act.tanh,
                    activation_options=Act.tanh,
                    aggregation_default=Agg.sum,
                    aggregation_options=Agg.sum,
                ),
                output_transform=Act.standard_sigmoid,  # the activation function for output node
                mutation=DefaultMutation(
                    node_add=0.1,
                    conn_add=0.1,
                    node_delete=0,
                    conn_delete=0,
                ),
            ),
        ),
        problem=XOR3d(),
        generation_limit=500,
        fitness_target=-1e-8,
    )

    # initialize state
    state = pipeline.setup()
    # print(state)
    # run until terminate
    state, best = pipeline.auto_run(state)
    # show result
    pipeline.show(state, best)
