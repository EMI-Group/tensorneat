import evox
import jax
from jax import jit, vmap, numpy as jnp

from configs import Configer
from algorithms.neat import create_forward_function, topological_sort, unflatten_connections
from evox_adaptor import NEAT, Gym

if __name__ == '__main__':
    batch_policy = True
    key = jax.random.PRNGKey(42)

    monitor = evox.monitors.StdSOMonitor()
    neat_config = Configer.load_config('acrobot.ini')
    origin_forward_func = create_forward_function(neat_config)


    def neat_transform(pop):
        P = neat_config['pop_size']
        N = neat_config['maximum_nodes']
        C = neat_config['maximum_connections']

        pop_nodes = pop[:P * N * 5].reshape((P, N, 5))
        pop_cons = pop[P * N * 5:].reshape((P, C, 4))

        u_pop_cons = vmap(unflatten_connections)(pop_nodes, pop_cons)
        pop_seqs = vmap(topological_sort)(pop_nodes, u_pop_cons)
        return pop_seqs, pop_nodes, u_pop_cons

    # special policy for mountain car
    def neat_forward(genome, x):
        res = origin_forward_func(x, *genome)
        out = jnp.argmax(res)  # {0, 1, 2}
        return out


    forward_func = lambda pop, x: origin_forward_func(x, *pop)

    problem = Gym(
        policy=jit(vmap(neat_forward)),
        env_name="Acrobot-v1",
        pop_size=100,
    )

    # create a pipeline
    pipeline = evox.pipelines.StdPipeline(
        algorithm=NEAT(neat_config),
        problem=problem,
        pop_transform=jit(neat_transform),
        fitness_transform=monitor.record_fit,
    )
    # init the pipeline
    state = pipeline.init(key)

    # run the pipeline for 10 steps
    for i in range(30):
        state = pipeline.step(state)
        print(i, monitor.get_min_fitness())

    # obtain -62.0
    min_fitness = monitor.get_min_fitness()
    print(min_fitness)
