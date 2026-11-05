"""Solve the 3-input XOR3d problem (8 samples) with NEAT.

Fitness is maximised, so the shipped target `-1e-3` is "error below 1e-3" -- the
same convention as `xor.py`. A `-1e-4` bound is not reachable inside the demo
budget: measured at seed 42, letting the full 150 generations run ends at loss
`1.6e-04`, while the `-1e-3` target triggers at generation 144.

Run: python examples/func_fit/xor3d.py
"""

from tensorneat import Pipeline, algorithm, genome, problem
from tensorneat.common import ACT

if __name__ == "__main__":
    # Demo budget: pop 400 for at most 150 generations; seed 42 reaches the
    # -1e-3 target at generation 144 in ~20 s. A real run uses pop_size in the
    # thousands and more generations.
    pipeline = Pipeline(
        algorithm=algorithm.NEAT(
            pop_size=400,
            species_size=20,
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
        generation_limit=150,
        # fitness is maximised: this target is "error below 1e-3"
        fitness_target=-1e-3,
    )

    # initialize the host state; the evoxc graphs are built on the first step
    state = pipeline.setup()
    # run until the fitness target is reached or the generation limit hits
    state, best = pipeline.auto_run(state)
    # show the best individual found
    pipeline.show(state, best)
