"""
contains operations on the population: creating the next generation and population speciation.
"""
from functools import partial

import jax
import jax.numpy as jnp
from jax import jit, vmap

from jax import Array

from .genome import distance, mutate, crossover
from .genome.utils import I_INT, fetch_first


@jit
def create_next_generation_then_speciate(rand_key, pop_nodes, pop_cons, winner, loser, elite_mask, new_node_keys,
                                         center_nodes, center_cons, species_keys, new_species_key_start,
                                         jit_config):
    # create next generation
    pop_nodes, pop_cons = create_next_generation(rand_key, pop_nodes, pop_cons, winner, loser, elite_mask,
                                                 new_node_keys, jit_config)

    # speciate
    idx2specie, spe_center_nodes, spe_center_cons, species_keys = \
        speciate(pop_nodes, pop_cons, center_nodes, center_cons, species_keys, new_species_key_start, jit_config)

    return pop_nodes, pop_cons, idx2specie, spe_center_nodes, spe_center_cons, species_keys


@jit
def create_next_generation(rand_key, pop_nodes, pop_cons, winner, loser, elite_mask, new_node_keys, jit_config):
    # prepare random keys
    pop_size = pop_nodes.shape[0]
    k1, k2 = jax.random.split(rand_key, 2)
    crossover_rand_keys = jax.random.split(k1, pop_size)
    mutate_rand_keys = jax.random.split(k2, pop_size)

    # batch crossover
    wpn, wpc = pop_nodes[winner], pop_cons[winner]  # winner pop nodes, winner pop connections
    lpn, lpc = pop_nodes[loser], pop_cons[loser]  # loser pop nodes, loser pop connections
    npn, npc = vmap(crossover)(crossover_rand_keys, wpn, wpc, lpn, lpc)  # new pop nodes, new pop connections

    # batch mutation
    mutate_func = vmap(mutate, in_axes=(0, 0, 0, 0, None))
    m_npn, m_npc = mutate_func(mutate_rand_keys, npn, npc, new_node_keys, jit_config)  # mutate_new_pop_nodes

    # elitism don't mutate
    pop_nodes = jnp.where(elite_mask[:, None, None], npn, m_npn)
    pop_cons = jnp.where(elite_mask[:, None, None], npc, m_npc)

    return pop_nodes, pop_cons


@jit
def speciate(pop_nodes, pop_cons, center_nodes, center_cons, species_keys, new_species_key_start, jit_config):
    """
    args:
        pop_nodes: (pop_size, N, 5)
        pop_cons: (pop_size, C, 4)
        spe_center_nodes: (species_size, N, 5)
        spe_center_cons: (species_size, C, 4)
    """
    pop_size, species_size = pop_nodes.shape[0], center_nodes.shape[0]

    # prepare distance functions
    o2p_distance_func = vmap(distance, in_axes=(None, None, 0, 0, None))  # one to population
    s2p_distance_func = vmap(
        o2p_distance_func, in_axes=(0, 0, None, None, None)  # center to population
    )

    # idx to specie key
    idx2specie = jnp.full((pop_size,), I_INT, dtype=jnp.int32)  # I_INT means not assigned to any species

    # part 1: find new centers
    # the distance between each species' center and each genome in population
    s2p_distance = s2p_distance_func(center_nodes, center_cons, pop_nodes, pop_cons, jit_config)

    def find_new_centers(i, carry):
        i2s, cn, cc = carry
        # find new center
        idx = argmin_with_mask(s2p_distance[i], mask=i2s == I_INT)

        # check species[i] exist or not
        # if not exist, set idx and i to I_INT, jax will not do array value assignment
        idx = jnp.where(species_keys[i] != I_INT, idx, I_INT)
        i = jnp.where(species_keys[i] != I_INT, i, I_INT)

        i2s = i2s.at[idx].set(species_keys[i])
        cn = cn.at[i].set(pop_nodes[idx])
        cc = cc.at[i].set(pop_cons[idx])
        return i2s, cn, cc

    idx2specie, center_nodes, center_cons = \
        jax.lax.fori_loop(0, species_size, find_new_centers, (idx2specie, center_nodes, center_cons))

    # part 2: assign members to each species
    def cond_func(carry):
        i, i2s, cn, cc, sk, ck = carry  # sk is short for species_keys, ck is short for current key
        not_all_assigned = ~jnp.all(i2s != I_INT)
        not_reach_species_upper_bounds = i < species_size
        return not_all_assigned & not_reach_species_upper_bounds

    def body_func(carry):
        i, i2s, cn, cc, sk, ck = carry  # scn is short for spe_center_nodes, scc is short for spe_center_cons

        i2s, scn, scc, sk, ck = jax.lax.cond(
            sk[i] == I_INT,  # whether the current species is existing or not
            create_new_specie,  # if not existing, create a new specie
            update_exist_specie,  # if existing, update the specie
            (i, i2s, cn, cc, sk, ck)
        )

        return i + 1, i2s, scn, scc, sk, ck

    def create_new_specie(carry):
        i, i2s, cn, cc, sk, ck = carry

        # pick the first one who has not been assigned to any species
        idx = fetch_first(i2s == I_INT)

        # assign it to the new species
        sk = sk.at[i].set(ck)
        i2s = i2s.at[idx].set(ck)

        # update center genomes
        cn = cn.at[i].set(pop_nodes[idx])
        cc = cc.at[i].set(pop_cons[idx])

        i2s = speciate_by_threshold((i, i2s, cn, cc, sk))
        return i2s, cn, cc, sk, ck + 1  # change to next new speciate key

    def update_exist_specie(carry):
        i, i2s, cn, cc, sk, ck = carry

        i2s = speciate_by_threshold((i, i2s, cn, cc, sk))

        return i2s, cn, cc, sk, ck

    def speciate_by_threshold(carry):
        i, i2s, cn, cc, sk = carry

        # distance between such center genome and ppo genomes
        o2p_distance = o2p_distance_func(cn[i], cc[i], pop_nodes, pop_cons, jit_config)
        close_enough_mask = o2p_distance < jit_config['compatibility_threshold']

        # when it is close enough, assign it to the species, remember not to update genome has already been assigned
        i2s = jnp.where(close_enough_mask & (i2s == I_INT), sk[i], i2s)
        return i2s

    current_new_key = new_species_key_start

    # update idx2specie
    _, idx2specie, center_nodes, center_cons, species_keys, _ = jax.lax.while_loop(
        cond_func,
        body_func,
        (0, idx2specie, center_nodes, center_cons, species_keys, current_new_key)
    )

    # if there are still some pop genomes not assigned to any species, add them to the last genome
    # this condition seems to be only happened when the number of species is reached species upper bounds
    idx2specie = jnp.where(idx2specie == I_INT, species_keys[-1], idx2specie)

    return idx2specie, center_nodes, center_cons, species_keys


@jit
def argmin_with_mask(arr: Array, mask: Array) -> Array:
    masked_arr = jnp.where(mask, arr, jnp.inf)
    min_idx = jnp.argmin(masked_arr)
    return min_idx
