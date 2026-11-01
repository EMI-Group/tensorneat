"""Fit the Pagie polynomial with NEAT over a sampled 2-D box.

The target is ``f(x, y) = 1 / (1 + x**-4) + 1 / (1 + y**-4)`` on ``[-1, 1]^2``.

Run: python examples/func_fit/custom_func_fit.py
"""

import numpy as np

from tensorneat import Pipeline, algorithm, genome, problem
from tensorneat.common import ACT, AGG


# `func` runs on the host with numpy, never inside the evoxc graph: the problem
# calls it once per sample point to build the (S, I) inputs / (S, O) targets.
def pagie_polynomial(inputs):
    """Map one 2-D point to the Pagie polynomial value."""
    x, y = inputs
    return 1 / (1 + x**-4) + 1 / (1 + y**-4)


if __name__ == "__main__":
    pipeline = Pipeline(
        algorithm=algorithm.NEAT(
            pop_size=1000,
            species_size=10,
            survival_threshold=0.01,
            genome=genome.DefaultGenome(
                num_inputs=2,
                num_outputs=1,
                init_hidden_layers=(),
                # a bias node plus no explicit `response` column, and the
                # activation/aggregation menus the Pagie polynomial needs
                node_gene=genome.BiasNode(
                    activation_options=[ACT.identity, ACT.inv, ACT.sin],
                    aggregation_options=[AGG.sum, AGG.product],
                ),
                output_transform=ACT.identity,
            ),
        ),
        problem=problem.CustomFuncFit(
            func=pagie_polynomial,
            low_bounds=np.array([-1.0, -1.0]),
            upper_bounds=np.array([1.0, 1.0]),
            method="sample",
            num_samples=100,
            seed=42,
        ),
        seed=42,
        generation_limit=100,
        # fitness is maximised: this target is "error below 1e-2"
        fitness_target=-1e-2,
    )

    # initialize the state; this is where the evoxc graphs are built
    state = pipeline.setup()
    # run until the fitness target is reached or the generation limit hits
    state, best = pipeline.auto_run(state)
    # show the best individual found
    pipeline.show(state, best)
