import jax.numpy as jnp

from tensorneat.pipeline import Pipeline
from tensorneat.algorithm.neat import NEAT
from tensorneat.algorithm.hyperneat import HyperNEAT, FullSubstrate
from tensorneat.genome import DefaultGenome
from tensorneat.common import Act

from tensorneat.problem import GymNaxEnv

if __name__ == "__main__":

    # the num of input_coors is 5
    # 4 is for cartpole inputs, 1 is for bias
    pipeline = Pipeline(
        algorithm=HyperNEAT(
            substrate=FullSubstrate(
                input_coors=((-1, -1), (-0.5, -1), (0, -1), (0.5, -1), (1, -1)),
                hidden_coors=((-1, 0), (0, 0), (1, 0)),
                output_coors=((-1, 1), (1, 1)),
            ),
            neat=NEAT(
                pop_size=10000,
                species_size=20,
                survival_threshold=0.01,
                genome=DefaultGenome(
                    num_inputs=4,  # size of query coors
                    num_outputs=1,
                    init_hidden_layers=(),
                    output_transform=Act.standard_tanh,
                ),
            ),
            activation=Act.tanh,
            activate_time=10,
            output_transform=jnp.argmax,
        ),
        problem=GymNaxEnv(
            env_name="CartPole-v1",
            repeat_times=5,
        ),
        generation_limit=300,
        fitness_target=-1e-6,
    )

    # initialize state
    state = pipeline.setup()
    # print(state)
    # run until terminate
    state, best = pipeline.auto_run(state)
