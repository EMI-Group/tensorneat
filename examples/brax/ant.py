
from tensorneat.pipeline import Pipeline
from tensorneat.algorithm.neat import NEAT
from tensorneat.genome import DefaultGenome, BiasNode, DefaultConn,DefaultMutation

from tensorneat.problem.rl import BraxEnv
from tensorneat.common import ACT, AGG
import jax
def random_sample_policy(randkey, obs):
    return jax.random.uniform(randkey, (8,), minval=-1.0, maxval=1.0)
if __name__ == "__main__":
    pipeline = Pipeline(
        algorithm=NEAT(
            pop_size=3000,
            species_size=20,
            survival_threshold=0.1,
            compatibility_threshold=0.8,
            genome=DefaultGenome(
                max_nodes=100,
                max_conns=1500,
                num_inputs=27,
                num_outputs=8,
                init_hidden_layers=(30,),
                mutation=DefaultMutation(
                    node_delete=0.0,
                ),
                node_gene=BiasNode(
                    bias_init_std=0.1,
                    bias_mutate_power=0.05,
                    bias_mutate_rate=0.01,
                    bias_replace_rate=0.0,
                    activation_options=ACT.tanh,
                    aggregation_options=AGG.sum,
                ),
                conn_gene=DefaultConn(
                    weight_init_mean=0.0,
                    weight_init_std=0.1,
                    weight_mutate_power=0.05,
                    weight_replace_rate=0.0,
                    weight_mutate_rate=0.001,
                ),
                output_transform=ACT.tanh,
            ),
        ),
        problem=BraxEnv(
            env_name="ant",
            max_step=1000,
        ),
        seed=42,
        generation_limit=100,
        fitness_target=8000,
    )

    # initialize state
    state = pipeline.setup()
    # print(state)
    # run until terminate
    state, best = pipeline.auto_run(state)