"""Solve the 2-input XOR problem with NEAT.

Run: python examples/func_fit/xor.py
"""

from tensorneat import Pipeline, algorithm, genome, problem
from tensorneat.common import ACT

if __name__ == "__main__":
    # Demo budget: pop 200 for at most 150 generations (it converges around
    # generation 75); a real run uses pop_size in the thousands and more generations.
    pipeline = Pipeline(
        algorithm=algorithm.NEAT(
            pop_size=200,
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
        generation_limit=150,
        # fitness is maximised: this target is "error below 1e-3"
        fitness_target=-1e-3,
    )

    # host-only setup: random key, initial population, task data -- no graph yet
    state = pipeline.setup()
    # evolve: the evoxc graph is built on the first step, then cached and reused
    state, best = pipeline.auto_run(state)
    # show the best individual found
    pipeline.show(state, best)
    print(
        f"final loss: {-pipeline.best_fitness:.2e}"
        f" after {len(pipeline.fitness_history)} generations"
    )
