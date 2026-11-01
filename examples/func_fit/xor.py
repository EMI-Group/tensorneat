"""Solve the 2-input XOR problem with NEAT.

Run: python examples/func_fit/xor.py
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
                num_inputs=2,
                num_outputs=1,
                max_nodes=12,
                output_transform=ACT.sigmoid,
            ),
        ),
        problem=problem.XOR(),
        seed=42,
        generation_limit=40,
        # fitness is maximised: this target is "error below 1e-3"
        fitness_target=-1e-3,
    )

    # initialize the state; this is where the evoxc graphs are built
    state = pipeline.setup()
    # run until the fitness target is reached or the generation limit hits
    state, best = pipeline.auto_run(state)
    # show the best individual found
    pipeline.show(state, best)
