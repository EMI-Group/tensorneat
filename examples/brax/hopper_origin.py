from tensorneat.pipeline import Pipeline
from tensorneat.algorithm.neat import NEAT
from tensorneat.genome import DefaultGenome, OriginNode, OriginConn

from tensorneat.problem.rl import BraxEnv
from tensorneat.common import ACT, AGG

"""
Solving Hopper with OriginGene
See https://github.com/EMI-Group/tensorneat/issues/11
"""

if __name__ == "__main__":
    pipeline = Pipeline(
        algorithm=NEAT(
            pop_size=1000,
            species_size=20,
            survival_threshold=0.1,
            compatibility_threshold=1.0,
            genome=DefaultGenome(
                num_inputs=11,
                num_outputs=3,
                init_hidden_layers=(),
                # origin node gene, which use the same crossover behavior to the origin NEAT paper
                node_gene=OriginNode(
                    activation_options=ACT.tanh,
                    aggregation_options=AGG.sum,
                    response_lower_bound = 1,
                    response_upper_bound= 1,  # fix response to 1
                ),
                # use origin connection, which using historical marker
                conn_gene=OriginConn(),  
                output_transform=ACT.tanh,
            ),
        ),
        problem=BraxEnv(
            env_name="hopper",
            max_step=1000,
        ),
        seed=42,
        generation_limit=100,
        fitness_target=5000,
    )

    # initialize state
    state = pipeline.setup()
    # print(state)
    # run until terminate
    state, best = pipeline.auto_run(state)
