import jax
import jax.numpy as jnp

from evox import workflows, problems

from tensorneat.common.evox_adaptors import EvoXAlgorithmAdaptor, TensorNEATMonitor
from tensorneat.algorithm import NEAT
from tensorneat.genome import DefaultGenome, BiasNode
from tensorneat.common import ACT, AGG

neat_algorithm = NEAT(
    pop_size=1000,
    species_size=20,
    survival_threshold=0.1,
    compatibility_threshold=1.0,
    genome=DefaultGenome(
        max_nodes=50,
        max_conns=200,
        num_inputs=17,
        num_outputs=6,
        init_hidden_layers=(),
        node_gene=BiasNode(
            activation_options=ACT.tanh,
            aggregation_options=AGG.sum,
        ),
        output_transform=ACT.tanh,
    ),
)
evox_algorithm = EvoXAlgorithmAdaptor(neat_algorithm)

key = jax.random.PRNGKey(42)
model_key, workflow_key = jax.random.split(key)

monitor = TensorNEATMonitor(neat_algorithm, is_save=False)
problem = problems.neuroevolution.Brax(
    env_name="walker2d",
    policy=evox_algorithm.forward,
    max_episode_length=1000,
    num_episodes=1,
)


def nan2inf(x):
    return jnp.where(jnp.isnan(x), -jnp.inf, x)


# create a workflow
workflow = workflows.StdWorkflow(
    algorithm=evox_algorithm,
    problem=problem,
    candidate_transforms=[jax.jit(jax.vmap(evox_algorithm.transform))],
    fitness_transforms=[nan2inf],
    monitors=[monitor],
    opt_direction="max",
)

# init the workflow
state = workflow.init(workflow_key)

# enable multi devices
state = workflow.enable_multi_devices(state)

# run the workflow for 100 steps
for i in range(100):
    train_info, state = workflow.step(state)
    monitor.show()
