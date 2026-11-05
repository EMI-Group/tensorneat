"""Solve XOR3d with a recurrent genome (recurrent connections allowed).

Run: python examples/func_fit/xor_recurrent.py
"""

from tensorneat import Pipeline, algorithm, genome, problem
from tensorneat.common import ACT

if __name__ == "__main__":
    # Demo budget: pop 300 for at most 200 generations (seed 42 reaches the
    # fitness target at generation 72); a real run uses pop_size in the
    # thousands and a few hundred generations.
    pipeline = Pipeline(
        algorithm=algorithm.NEAT(
            pop_size=300,
            species_size=10,
            # 0.01 floors to one parent per species, which does leave the ~0.25
            # plateau but climbs too slowly to reach the target in 200
            # generations (loss ~9e-04 at generation 200); 0.3 breeds from
            # several parents per species and reaches -1e-4 at generation 72
            survival_threshold=0.3,
            genome=genome.RecurrentGenome(
                num_inputs=3,
                num_outputs=1,
                max_nodes=16,
                # a recurrent network can accumulate many incoming edges per node,
                # and the in-degree window is a static: size it generously or the
                # run stops with "the execution schedule dropped N connection(s)"
                max_in_degree=16,
                # a recurrent genome is tick-unrolled, so `activate_time` is the
                # cost knob: the forward pass runs that many relaxation ticks
                activate_time=10,
                output_transform=ACT.sigmoid,
            ),
        ),
        problem=problem.XOR3d(),
        seed=42,
        generation_limit=200,
        # fitness is maximised: this target is "error below 1e-4"
        fitness_target=-1e-4,
    )

    # initialize the host state; the evoxc graphs are built on the first step
    state = pipeline.setup()
    # run until the fitness target is reached or the generation limit hits
    state, best = pipeline.auto_run(state)
    # show the best individual found
    pipeline.show(state, best)
