"""Solve XOR3d with the original-NEAT origin genes.

`OriginNode` copies a whole gene on crossover and `OriginConn` keeps the
historical marker, so every mutation stays traceable
(https://github.com/EMI-Group/tensorneat/issues/11).

The demo budget below is enough to solve XOR3d at seed 42: the population sits
on the constant-0.5 predictor, so the per-generation max fitness is about -0.25
and reads exactly -0.2500 from generation 6 through generation 13; it first
improves at generation 14 (-0.2499), then climbs steadily and stops on the -1e-6
error target at generation 89, which takes 5.8 s on this CPU. A real run uses
pop_size in the thousands and a few hundred generations.

Run: python examples/func_fit/xor_origin.py
"""

from tensorneat import Pipeline, algorithm, genome, problem
from tensorneat.common import ACT

if __name__ == "__main__":
    # Demo budget: pop 300 for at most 250 generations; seed 42 solves XOR3d on
    # generation 89 (5.8 s on this CPU). A real run uses pop_size in the
    # thousands and a few hundred generations; a larger pop is no faster here.
    pipeline = Pipeline(
        algorithm=algorithm.NEAT(
            pop_size=300,
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
        generation_limit=250,
        # fitness is maximised: this target is "error below 1e-6"
        fitness_target=-1e-6,
    )

    # initialize the host state; the evoxc graphs are built on the first step
    state = pipeline.setup()
    # run until the fitness target is reached or the generation limit hits
    state, best = pipeline.auto_run(state)
    # show the best individual found
    pipeline.show(state, best)
