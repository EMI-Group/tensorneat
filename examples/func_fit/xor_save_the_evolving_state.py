import jax
import numpy as np
from tensorneat.pipeline import Pipeline
from tensorneat import algorithm, genome, problem
from tensorneat.common import ACT

# neccessary settings
algorithm = algorithm.NEAT(
    pop_size=1000,
    species_size=20,
    survival_threshold=0.01,
    genome=genome.DefaultGenome(
        num_inputs=3,
        num_outputs=1,
        max_nodes=7,
        output_transform=ACT.sigmoid,
    ),
)
problem = problem.XOR3d()

pipeline = Pipeline(
    algorithm,
    problem,
    generation_limit=200,  # actually useless when we don't using auto_run()
    fitness_target=-1e-6,  # actually useless when we don't using auto_run()
    seed=42,
)
state = pipeline.setup()

# compile step to speed up
compiled_step = jax.jit(pipeline.step).lower(state).compile()

current_generation = 0
# run 50 generations
for i in range(50):
    state, previous_pop, fitnesses = compiled_step(state)
    fitnesses = jax.device_get(fitnesses)  # move fitness from gpu to cpu for printing
    print(f"Generation {current_generation}, best fitness: {max(fitnesses)}")
    current_generation += 1

# obtain the best individual
best_idx = np.argmax(fitnesses)
best_nodes, best_conns = previous_pop[0][best_idx], previous_pop[1][best_idx]
# make it inference
transformed = algorithm.genome.transform(state, best_nodes, best_conns)
xor3d_outputs = jax.vmap(algorithm.genome.forward, in_axes=(None, None, 0))(state, transformed, problem.inputs)
print(f"{xor3d_outputs=}")

# save the evolving state
state.save("./evolving_state.pkl")
print("save the evolving state to ./evolving_state.pkl")