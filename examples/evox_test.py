import jax
from jax import numpy as jnp

from evox import algorithms, problems, pipelines
from evox.monitors import StdSOMonitor

monitor = StdSOMonitor()

pso = algorithms.PSO(
    lb=jnp.full(shape=(2,), fill_value=-32),
    ub=jnp.full(shape=(2,), fill_value=32),
    pop_size=100,
)

ackley = problems.classic.Ackley()

pipeline = pipelines.StdPipeline(pso, ackley, fitness_transform=monitor.record_fit)

key = jax.random.PRNGKey(42)
state = pipeline.init(key)

# run the pipeline for 100 steps
for i in range(100):
    state = pipeline.step(state)

print(monitor.get_min_fitness())