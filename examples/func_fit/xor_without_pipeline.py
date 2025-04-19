import jax, jax.numpy as jnp
import numpy as np
from tensorneat import algorithm, genome, problem
from tensorneat.common import ACT, State

POPSIZE = 10000

# STEP 0: create initial state which contains randkey
state = State(
    randkey = jax.random.key(42)
)

# STEP 1: define the NEAT algorithm and prepare necessary functions
algorithm = algorithm.NEAT(
    pop_size=POPSIZE,
    species_size=20,
    survival_threshold=0.01,
    genome=genome.DefaultGenome(
        num_inputs=3,
        num_outputs=1,
        max_nodes=7,
        output_transform=ACT.sigmoid,
    ),
)
population_transform = jax.vmap(algorithm.transform, in_axes=(None, 0))
jit_population_transform = jax.jit(population_transform)
jit_algorithm_tell = jax.jit(algorithm.tell)

state = algorithm.setup(state)  # setup algorithm and then save infos in previous created state


# STEP 2: define the function that return the fitness of population
problem = problem.XOR3d()  # problem that used in pipeline
state = problem.setup(state)  # setup algorithm and then save infos in previous created state

# function for evaluate a single network
def fitness_single(state, randkey, transformed):
    return problem.evaluate(state, randkey, algorithm.forward, transformed)

# funciton for evalute the population
fitness_population = jax.vmap(fitness_single, in_axes=(None, 0, 0))
jit_fitness_population = jax.jit(fitness_population)


# STEP 3: Run NEAT algorithm to solve the problem
while True:
    population = algorithm.ask(state)
    # network in TensorNEAT need to be transformed before calculation
    pop_transformed = jit_population_transform(state, population)

    randkeys = jax.random.split(state.randkey, POPSIZE)
    # evaluate networks and obtain their fitness
    pop_fitness = jit_fitness_population(state, randkeys, pop_transformed)
    # replace nan with -inf, necessary
    pop_fitness = jnp.where(jnp.isnan(pop_fitness), -jnp.inf, pop_fitness)

    # Do whatever you want here
    cpu_pop_fitness = jax.device_get(pop_fitness) 
    best_idx = np.argmax(cpu_pop_fitness)
    print(f"best fitness: {cpu_pop_fitness[best_idx]}")

    if cpu_pop_fitness[best_idx] > -1e-6:  # stop check
        best = (
            population[0][best_idx],
            population[1][best_idx],
        )  # population = (pop_nodes, pop_conns)
        break

    # tell the fitness to algorithm and update population
    state = jit_algorithm_tell(state, pop_fitness)


# STEP 4: Do anything you want to the best network
network = algorithm.genome.network_dict(state, *best)
print(algorithm.genome.repr(state, *best))

# validate the output for the best
best_transformed = algorithm.transform(state, best)
output = algorithm.forward(state, best_transformed, jnp.array([1, 0, 1]))
print(f"{output=}")






