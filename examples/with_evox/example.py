import jax
import jax.numpy as jnp

from evox import workflows, algorithms, problems

from tensorneat.examples.with_evox.evox_algorithm_adaptor import EvoXAlgorithmAdaptor
from tensorneat.examples.with_evox.tensorneat_monitor import TensorNEATMonitor
from tensorneat.algorithm import NEAT
from tensorneat.algorithm.neat import DefaultSpecies, DefaultGenome, DefaultNodeGene
from tensorneat.common import ACT

neat_algorithm = NEAT(
    species=DefaultSpecies(
        genome=DefaultGenome(
            num_inputs=17,
            num_outputs=6,
            max_nodes=200,
            max_conns=500,
            node_gene=DefaultNodeGene(
                activation_options=(ACT.standard_tanh,),
                activation_default=ACT.standard_tanh,
            ),
            output_transform=ACT.tanh,
        ),
        pop_size=10000,
        species_size=10,
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
    backend="mjx"
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
# state = workflow.enable_multi_devices(state)
# run the workflow for 100 steps
import time

for i in range(100):
    tic = time.time()
    train_info, state = workflow.step(state)
    monitor.show()