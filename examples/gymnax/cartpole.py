import jax.numpy as jnp

from tensorneat.pipeline import Pipeline
from tensorneat.algorithm.neat import NEAT
from tensorneat.genome import DefaultGenome, BiasNode

from tensorneat.problem.rl import GymNaxEnv
from tensorneat.common import Act, Agg



if __name__ == "__main__":
    # the network has 2 outputs, the max one will be the action
    # as the action of cartpole is {0, 1}

    pipeline = Pipeline(
        algorithm=NEAT(
            pop_size=1000,
            species_size=20,
            survival_threshold=0.1,
            compatibility_threshold=1.0,
            genome=DefaultGenome(
                num_inputs=4,
                num_outputs=2,
                init_hidden_layers=(),
                node_gene=BiasNode(
                    activation_options=Act.tanh,
                    aggregation_options=Agg.sum,
                ),
                output_transform=jnp.argmax,
            ),
        ),
        problem=GymNaxEnv(
            env_name="CartPole-v1",
            repeat_times=5,
        ),
        seed=42,
        generation_limit=100,
        fitness_target=500,
    )

    # initialize state
    state = pipeline.setup()

    # run until terminate
    state, best = pipeline.auto_run(state)
