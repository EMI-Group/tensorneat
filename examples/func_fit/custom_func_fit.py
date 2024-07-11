import jax.numpy as jnp

from tensorneat.pipeline import Pipeline
from tensorneat.algorithm.neat import NEAT
from tensorneat.genome import DefaultGenome, DefaultNode, DefaultMutation, BiasNode
from tensorneat.problem.func_fit import CustomFuncFit
from tensorneat.common import ACT, AGG


def pagie_polynomial(inputs):
    x, y = inputs
    res = 1 / (1 + jnp.pow(x, -4)) + 1 / (1 + jnp.pow(y, -4))

    # important! returns an array, NOT a scalar
    return jnp.array([res])


if __name__ == "__main__":

    custom_problem = CustomFuncFit(
        func=pagie_polynomial,
        low_bounds=[-1, -1],
        upper_bounds=[1, 1],
        method="sample",
        num_samples=100,
    )

    pipeline = Pipeline(
        algorithm=NEAT(
            pop_size=10000,
            species_size=20,
            survival_threshold=0.01,
            genome=DefaultGenome(
                num_inputs=2,
                num_outputs=1,
                init_hidden_layers=(),
                node_gene=BiasNode(
                    activation_options=[ACT.identity, ACT.inv, ACT.square],
                    aggregation_options=[AGG.sum, AGG.product],
                ),
                output_transform=ACT.identity,
            ),
        ),
        problem=custom_problem,
        generation_limit=50,
        fitness_target=-1e-4,
        seed=42,
    )

    # initialize state
    state = pipeline.setup()
    # run until terminate
    state, best = pipeline.auto_run(state)
    # show result
    # pipeline.show(state, best)
    print(pipeline.algorithm.genome.repr(state, *best))
