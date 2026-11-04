"""Solve the 3-input XOR3d problem (8 samples) with NEAT.

Run: python examples/func_fit/xor3d.py
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
                num_inputs=3,
                num_outputs=1,
                max_nodes=16,
                output_transform=ACT.sigmoid,
            ),
        ),
        problem=problem.XOR3d(),
        seed=42,
        generation_limit=50,
        # fitness is maximised: this target is "error below 1e-4"
        fitness_target=-1e-4,
    )

    # initialize the host state; the evoxc graphs are built on the first step
    state = pipeline.setup()
    # run until the fitness target is reached or the generation limit hits
    state, best = pipeline.auto_run(state)
    # show the best individual found
    pipeline.show(state, best)
