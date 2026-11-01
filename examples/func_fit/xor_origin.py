"""Solve XOR3d with the original-NEAT origin genes.

`OriginNode` copies a whole gene on crossover and `OriginConn` keeps the
historical marker, so every mutation stays traceable
(https://github.com/EMI-Group/tensorneat/issues/11).

Run: python examples/func_fit/xor_origin.py
"""

from tensorneat import Pipeline, algorithm, genome, problem
from tensorneat.common import ACT

if __name__ == "__main__":
    # Demo budget: a real run uses pop_size in the thousands and a few hundred generations.
    pipeline = Pipeline(
        algorithm=algorithm.NEAT(
            pop_size=500,
            species_size=10,
            survival_threshold=0.01,
            genome=genome.DefaultGenome(
                node_gene=genome.OriginNode(),
                conn_gene=genome.OriginConn(),
                num_inputs=3,
                num_outputs=1,
                max_nodes=16,
                output_transform=ACT.sigmoid,
            ),
        ),
        problem=problem.XOR3d(),
        seed=42,
        generation_limit=40,
        # fitness is maximised: this target is "error below 1e-4"
        fitness_target=-1e-4,
    )

    # initialize the state; this is where the evoxc graphs are built
    state = pipeline.setup()
    # run until the fitness target is reached or the generation limit hits
    state, best = pipeline.auto_run(state)
    # show the best individual found
    pipeline.show(state, best)
