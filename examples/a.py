import jax
from jax import numpy as jnp

from config import Config
from core import Genome

config = Config()
from dataclasses import asdict

print(asdict(config))

pop_nodes = jnp.ones((Config.basic.pop_size, Config.neat.maximum_nodes, 3))
pop_conns = jnp.ones((Config.basic.pop_size, Config.neat.maximum_conns, 5))

pop_genomes1 = jax.vmap(Genome)(pop_nodes, pop_conns)
pop_genomes2 = Genome(pop_nodes, pop_conns)

print(pop_genomes)
print(pop_genomes[0])

@jax.vmap
def pop_cnts(genome):
    return genome.count()

cnts = pop_cnts(pop_genomes)

print(cnts)

