from functools import partial

import jax
import jax.numpy as jnp
from jax import jit, vmap

from jax import Array

from .genome import distance, mutate, crossover
from .genome.utils import I_INT, fetch_first, argmin_with_mask


@jit
def create_next_generation_then_speciate(rand_key, pop_nodes, pop_cons, winner_part, loser_part, elite_mask,
                                         new_node_keys,
                                         pre_spe_center_nodes, pre_spe_center_cons, species_keys, new_species_key_start,
                                         species_kwargs, mutate_kwargs):
    # create next generation
    pop_nodes, pop_cons = create_next_generation(rand_key, pop_nodes, pop_cons, winner_part, loser_part, elite_mask,
                                                 new_node_keys, **mutate_kwargs)

    # speciate
    idx2specie, spe_center_nodes, spe_center_cons, species_keys = speciate(pop_nodes, pop_cons, pre_spe_center_nodes,
                                                                           pre_spe_center_cons, species_keys,
                                                                           new_species_key_start, **species_kwargs)

    return pop_nodes, pop_cons, idx2specie, spe_center_nodes, spe_center_cons, species_keys


@jit
def speciate(pop_nodes: Array, pop_cons: Array, spe_center_nodes: Array, spe_center_cons: Array,
             species_keys, new_species_key_start,
             disjoint_coe: float = 1., compatibility_coe: float = 0.5, compatibility_threshold=3.0
             ):
    """
    args:
        pop_nodes: (pop_size, N, 5)
        pop_cons: (pop_size, C, 4)
        spe_center_nodes: (species_size, N, 5)
        spe_center_cons: (species_size, C, 4)
    """
    pop_size, species_size = pop_nodes.shape[0], spe_center_nodes.shape[0]

    # prepare distance functions
    distance_with_args = partial(distance, disjoint_coe=disjoint_coe, compatibility_coe=compatibility_coe)
    o2p_distance_func = vmap(distance_with_args, in_axes=(None, None, 0, 0))
    s2p_distance_func = vmap(
        o2p_distance_func, in_axes=(0, 0, None, None)
    )

    # idx to specie key
    idx2specie = jnp.full((pop_size,), I_INT, dtype=jnp.int32)  # I_INT means not assigned to any species

    # part 1: find new centers
    # the distance between each species' center and each genome in population
    s2p_distance = s2p_distance_func(spe_center_nodes, spe_center_cons, pop_nodes, pop_cons)

    def find_new_centers(i, carry):
        i2s, scn, scc = carry
        # find new center
        idx = argmin_with_mask(s2p_distance[i], mask=i2s == I_INT)

        # check species[i] exist or not
        # if not exist, set idx and i to I_INT, jax will not do array value assignment
        idx = jnp.where(species_keys[i] != I_INT, idx, I_INT)
        i = jnp.where(species_keys[i] != I_INT, i, I_INT)

        i2s = i2s.at[idx].set(species_keys[i])
        scn = scn.at[i].set(pop_nodes[idx])
        scc = scc.at[i].set(pop_cons[idx])
        return i2s, scn, scc

    idx2specie, spe_center_nodes, spe_center_cons = jax.lax.fori_loop(0, species_size, find_new_centers, (idx2specie, spe_center_nodes, spe_center_cons))

    def continue_execute_while(carry):
        i, i2s, scn, scc, sk, ck = carry  # sk is short for species_keys, ck is short for current key
        not_all_assigned = ~jnp.all(i2s != I_INT)
        not_reach_species_upper_bounds = i < species_size
        return not_all_assigned & not_reach_species_upper_bounds

    def deal_with_each_center_genome(carry):
        i, i2s, scn, scc, sk, ck = carry  # scn is short for spe_center_nodes, scc is short for spe_center_cons
        center_nodes, center_cons = spe_center_nodes[i], spe_center_cons[i]

        i2s, scn, scc, sk, ck = jax.lax.cond(
            jnp.all(jnp.isnan(center_nodes)),  # whether the center genome is valid
            create_new_specie,  # if not valid, create a new specie
            update_exist_specie,  # if valid, update the specie
            (i, i2s, scn, scc, sk, ck)
        )

        return i + 1, i2s, scn, scc, sk, ck

    def create_new_specie(carry):
        i, i2s, scn, scc, sk, ck = carry
        # pick the first one who has not been assigned to any species
        idx = fetch_first(i2s == I_INT)

        # assign it to new specie
        sk = sk.at[i].set(ck)
        i2s = i2s.at[idx].set(ck)

        # update center genomes
        scn = scn.at[i].set(pop_nodes[idx])
        scc = scc.at[i].set(pop_cons[idx])

        i2s, scn, scc = speciate_by_threshold((i, i2s, scn, scc, sk))
        return i2s, scn, scc, sk, ck + 1  # change to next new speciate key

    def update_exist_specie(carry):
        i, i2s, scn, scc, sk, ck = carry

        i2s, scn, scc = speciate_by_threshold((i, i2s, scn, scc, sk))
        return i2s, scn, scc, sk, ck

    def speciate_by_threshold(carry):
        i, i2s, scn, scc, sk = carry
        # distance between such center genome and ppo genomes
        o2p_distance = o2p_distance_func(scn[i], scc[i], pop_nodes, pop_cons)
        close_enough_mask = o2p_distance < compatibility_threshold

        # when it is close enough, assign it to the species, remember not to update genome has already been assigned
        i2s = jnp.where(close_enough_mask & (i2s == I_INT), sk[i], i2s)
        return i2s, scn, scc

    current_new_key = new_species_key_start

    # update idx2specie
    _, idx2specie, spe_center_nodes, spe_center_cons, species_keys, new_species_key_start = jax.lax.while_loop(
        continue_execute_while,
        deal_with_each_center_genome,
        (0, idx2specie, spe_center_nodes, spe_center_cons, species_keys, current_new_key)
    )

    # if there are still some pop genomes not assigned to any species, add them to the last genome
    # this condition seems to be only happened when the number of species is reached species upper bounds
    idx2specie = jnp.where(idx2specie == I_INT, species_keys[-1], idx2specie)

    return idx2specie, spe_center_nodes, spe_center_cons, species_keys


@jit
def create_next_generation(rand_key, pop_nodes, pop_cons, winner_part, loser_part, elite_mask, new_node_keys,
                           **mutate_kwargs):
    # prepare functions
    batch_crossover = vmap(crossover)
    mutate_with_args = vmap(partial(mutate, **mutate_kwargs))

    pop_size = pop_nodes.shape[0]
    k1, k2 = jax.random.split(rand_key, 2)
    crossover_rand_keys = jax.random.split(k1, pop_size)
    mutate_rand_keys = jax.random.split(k2, pop_size)

    # batch crossover
    wpn = pop_nodes[winner_part]  # winner pop nodes
    wpc = pop_cons[winner_part]  # winner pop connections
    lpn = pop_nodes[loser_part]  # loser pop nodes
    lpc = pop_cons[loser_part]  # loser pop connections

    npn, npc = batch_crossover(crossover_rand_keys, wpn, wpc, lpn, lpc)  # new pop nodes, new pop connections

    m_npn, m_npc = mutate_with_args(mutate_rand_keys, npn, npc, new_node_keys)  # mutate_new_pop_nodes

    # elitism don't mutate
    pop_nodes = jnp.where(elite_mask[:, None, None], npn, m_npn)
    pop_cons = jnp.where(elite_mask[:, None, None], npc, m_npc)

    return pop_nodes, pop_cons
