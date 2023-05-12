from functools import partial

import jax
import jax.numpy as jnp
from jax import jit, vmap

from jax import Array

from .genome import distance
from .genome.utils import I_INT, fetch_first, argmin_with_mask


@jit
def jitable_speciate(pop_nodes: Array, pop_cons: Array, spe_center_nodes: Array, spe_center_cons: Array,
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

    idx2specie = jnp.full((pop_size,), I_INT, dtype=jnp.int32)  # I_INT means not assigned to any species

    # the distance between each species' center and each genome in population
    s2p_distance = s2p_distance_func(spe_center_nodes, spe_center_cons, pop_nodes, pop_cons)

    def continue_execute_while(carry):
        i, i2s, scn, scc = carry
        not_all_assigned = ~jnp.all(i2s != I_INT)
        not_reach_species_upper_bounds = i < species_size
        return not_all_assigned & not_reach_species_upper_bounds

    def deal_with_each_center_genome(carry):
        i, i2s, scn, scc = carry  # scn is short for spe_center_nodes, scc is short for spe_center_cons
        center_nodes, center_cons = spe_center_nodes[i], spe_center_cons[i]

        i2s, scn, scc = jax.lax.cond(
            jnp.all(jnp.isnan(center_nodes)),  # whether the center genome is valid
            create_new_specie,  # if not valid, create a new specie
            update_exist_specie,  # if valid, update the specie
            (i, i2s, scn, scc)
        )

        return i + 1, i2s, scn, scc

    def create_new_specie(carry):
        i, i2s, scn, scc = carry
        # pick the first one who has not been assigned to any species
        idx = fetch_first(i2s == I_INT)

        # assign it to new specie
        i2s = i2s.at[idx].set(i)

        # update center genomes
        scn = scn.at[i].set(pop_nodes[idx])
        scc = scc.at[i].set(pop_cons[idx])

        i2s, scn, scc = speciate_by_threshold((i, i2s, scn, scc))
        return i2s, scn, scc

    def update_exist_specie(carry):
        i, i2s, scn, scc = carry

        # find new center
        idx = argmin_with_mask(s2p_distance[i], mask=i2s == I_INT)

        # update new center
        i2s = i2s.at[idx].set(i)

        # update center genomes
        scn = scn.at[i].set(pop_nodes[idx])
        scc = scc.at[i].set(pop_cons[idx])

        i2s, scn, scc = speciate_by_threshold((i, i2s, scn, scc))
        return i2s, scn, scc

    def speciate_by_threshold(carry):
        i, i2s, scn, scc = carry
        # distance between such center genome and ppo genomes
        o2p_distance = o2p_distance_func(scn[i], scc[i], pop_nodes, pop_cons)
        close_enough_mask = o2p_distance < compatibility_threshold

        # when it is close enough, assign it to the species, remember not to update genome has already been assigned
        i2s = jnp.where(close_enough_mask & (i2s == I_INT), i, i2s)
        return i2s, scn, scc

    # update idx2specie
    _, idx2specie, spe_center_nodes, spe_center_cons = jax.lax.while_loop(
        continue_execute_while,
        deal_with_each_center_genome,
        (0, idx2specie, spe_center_nodes, spe_center_cons)
    )

    # if there are still some pop genomes not assigned to any species, add them to the last genome
    # this condition seems to be only happened when the number of species is reached species upper bounds
    idx2specie = jnp.where(idx2specie == I_INT, species_size - 1, idx2specie)

    return idx2specie, spe_center_nodes, spe_center_cons
