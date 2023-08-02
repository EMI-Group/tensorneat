import jax
from jax import numpy as jnp, vmap

from config import NeatConfig
from core import Genome, State, Gene
from .mutate import mutate
from .crossover import crossover


def create_next_generation(config: NeatConfig, gene: Gene, state: State, randkey, winner, loser, elite_mask):
    # prepare random keys
    pop_size = state.idx2species.shape[0]
    new_node_keys = jnp.arange(pop_size) + state.next_node_key

    k1, k2 = jax.random.split(randkey, 2)
    crossover_rand_keys = jax.random.split(k1, pop_size)
    mutate_rand_keys = jax.random.split(k2, pop_size)

    # batch crossover
    wpn, wpc = state.pop_genomes.nodes[winner], state.pop_genomes.conns[winner]
    lpn, lpc = state.pop_genomes.nodes[loser], state.pop_genomes.conns[loser]
    n_genomes = vmap(crossover)(crossover_rand_keys, Genome(wpn, wpc), Genome(lpn, lpc))

    # batch mutation
    mutate_func = vmap(mutate, in_axes=(None, None, None, 0, 0, 0))
    m_n_genomes = mutate_func(config, gene, state, mutate_rand_keys, n_genomes, new_node_keys)  # mutate_new_pop_nodes

    # elitism don't mutate
    pop_nodes = jnp.where(elite_mask[:, None, None], n_genomes.nodes, m_n_genomes.nodes)
    pop_conns = jnp.where(elite_mask[:, None, None], n_genomes.conns, m_n_genomes.conns)

    # update next node key
    all_nodes_keys = pop_nodes[:, :, 0]
    max_node_key = jnp.max(jnp.where(jnp.isnan(all_nodes_keys), -jnp.inf, all_nodes_keys))
    next_node_key = max_node_key + 1

    return state.update(
        pop_genomes=Genome(pop_nodes, pop_conns),
        next_node_key=next_node_key,
    )
