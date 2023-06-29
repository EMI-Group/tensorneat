"""
Contains operations on the population: creating the next generation and population speciation.
These im.....
"""

# TODO: Complete python doc

import jax
from jax import jit, vmap, Array, numpy as jnp

from .genome import distance, mutate, crossover, I_INT, fetch_first, rank_elements


@jit
def update_species(randkey, fitness, species_info, idx2species, center_nodes, center_cons, generation, jit_config):
    """
    args:
        randkey: random key
        fitness: Array[(pop_size,), float], the fitness of each individual
        species_keys: Array[(species_size, 4), float], the information of each species
            [species_key, best_score, last_update, members_count]
        idx2species: Array[(pop_size,), int], map the individual to its species
        center_nodes: Array[(species_size, N, 4), float], the center nodes of each species
        center_cons: Array[(species_size, C, 4), float], the center connections of each species
        generation: int, current generation
        jit_config: Dict, the configuration of jit functions
    """

    # update the fitness of each species
    species_fitness = update_species_fitness(species_info, idx2species, fitness)

    # stagnation species
    species_fitness, species_info, center_nodes, center_cons = \
        stagnation(species_fitness, species_info, center_nodes, center_cons, generation, jit_config)

    # sort species_info by their fitness. (push nan to the end)
    sort_indices = jnp.argsort(species_fitness)[::-1]
    species_info = species_info[sort_indices]
    center_nodes, center_cons = center_nodes[sort_indices], center_cons[sort_indices]

    # decide the number of members of each species by their fitness
    spawn_number = cal_spawn_numbers(species_info, jit_config)

    # crossover info
    winner, loser, elite_mask = \
        create_crossover_pair(randkey, species_info, idx2species, spawn_number, fitness, jit_config)

    return species_info, center_nodes, center_cons, winner, loser, elite_mask


def update_species_fitness(species_info, idx2species, fitness):
    """
    obtain the fitness of the species by the fitness of each individual.
    use max criterion.
    """

    def aux_func(idx):
        species_key = species_info[idx, 0]
        s_fitness = jnp.where(idx2species == species_key, fitness, -jnp.inf)
        f = jnp.max(s_fitness)
        return f

    return vmap(aux_func)(jnp.arange(species_info.shape[0]))


def stagnation(species_fitness, species_info, center_nodes, center_cons, generation, jit_config):
    """
    stagnation species.
    those species whose fitness is not better than the best fitness of the species for a long time will be stagnation.
    elitism species never stagnation
    """

    def aux_func(idx):
        s_fitness = species_fitness[idx]
        species_key, best_score, last_update, members_count = species_info[idx]
        st = (s_fitness <= best_score) & (generation - last_update > jit_config['max_stagnation'])
        last_update = jnp.where(s_fitness > best_score, generation, last_update)
        best_score = jnp.where(s_fitness > best_score, s_fitness, best_score)
        # stagnation condition
        return st, jnp.array([species_key, best_score, last_update, members_count])

    spe_st, species_info = vmap(aux_func)(jnp.arange(species_info.shape[0]))

    # elite species will not be stagnation
    species_rank = rank_elements(species_fitness)
    spe_st = jnp.where(species_rank < jit_config['species_elitism'], False, spe_st)  # elitism never stagnation

    # set stagnation species to nan
    species_info = jnp.where(spe_st[:, None], jnp.nan, species_info)
    center_nodes = jnp.where(spe_st[:, None, None], jnp.nan, center_nodes)
    center_cons = jnp.where(spe_st[:, None, None], jnp.nan, center_cons)
    species_fitness = jnp.where(spe_st, jnp.nan, species_fitness)

    return species_fitness, species_info, center_nodes, center_cons


def cal_spawn_numbers(species_info, jit_config):
    """
    decide the number of members of each species by their fitness rank.
    the species with higher fitness will have more members
    Linear ranking selection
        e.g. N = 3, P=10 -> probability = [0.5, 0.33, 0.17], spawn_number = [5, 3, 2]
    """

    is_species_valid = ~jnp.isnan(species_info[:, 0])
    valid_species_num = jnp.sum(is_species_valid)
    denominator = (valid_species_num + 1) * valid_species_num / 2  # obtain 3 + 2 + 1 = 6

    rank_score = valid_species_num - jnp.arange(species_info.shape[0])  # obtain [3, 2, 1]
    spawn_number_rate = rank_score / denominator  # obtain [0.5, 0.33, 0.17]
    spawn_number_rate = jnp.where(is_species_valid, spawn_number_rate, 0)  # set invalid species to 0

    spawn_number = jnp.floor(spawn_number_rate * jit_config['pop_size']).astype(jnp.int32)  # calculate member

    # must control the sum of spawn_number to be equal to pop_size
    error = jit_config['pop_size'] - jnp.sum(spawn_number)
    spawn_number = spawn_number.at[0].add(error)  # add error to the first species to control the sum of spawn_number

    return spawn_number


def create_crossover_pair(randkey, species_info, idx2species, spawn_number, fitness, jit_config):
    species_size = species_info.shape[0]
    pop_size = fitness.shape[0]
    s_idx = jnp.arange(species_size)
    p_idx = jnp.arange(pop_size)

    # def aux_func(key, idx):
    def aux_func(key, idx):
        members = idx2species == species_info[idx, 0]
        members_num = jnp.sum(members)

        members_fitness = jnp.where(members, fitness, -jnp.inf)
        sorted_member_indices = jnp.argsort(members_fitness)[::-1]

        elite_size = jit_config['genome_elitism']
        survive_size = jnp.floor(jit_config['survival_threshold'] * members_num).astype(jnp.int32)

        select_pro = (p_idx < survive_size) / survive_size
        fa, ma = jax.random.choice(key, sorted_member_indices, shape=(2, pop_size), replace=True, p=select_pro)

        # elite
        fa = jnp.where(p_idx < elite_size, sorted_member_indices, fa)
        ma = jnp.where(p_idx < elite_size, sorted_member_indices, ma)
        elite = jnp.where(p_idx < elite_size, True, False)
        return fa, ma, elite

    # fas, mas, elites = jax.lax.max(aux_func, (jax.random.split(randkey, species_size), s_idx))
    fas, mas, elites = vmap(aux_func)(jax.random.split(randkey, species_size), s_idx)

    spawn_number_cum = jnp.cumsum(spawn_number)

    def aux_func(idx):
        loc = jnp.argmax(idx < spawn_number_cum)

        # elite genomes are at the beginning of the species
        idx_in_species = jnp.where(loc > 0, idx - spawn_number_cum[loc - 1], idx)
        return fas[loc, idx_in_species], mas[loc, idx_in_species], elites[loc, idx_in_species]

    part1, part2, elite_mask = vmap(aux_func)(p_idx)

    is_part1_win = fitness[part1] >= fitness[part2]
    winner = jnp.where(is_part1_win, part1, part2)
    loser = jnp.where(is_part1_win, part2, part1)

    return winner, loser, elite_mask


@jit
def create_next_generation(rand_key, pop_nodes, pop_cons, winner, loser, elite_mask, generation, jit_config):
    # prepare random keys
    pop_size = pop_nodes.shape[0]
    new_node_keys = jnp.arange(pop_size) + generation * pop_size

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
def speciate(pop_nodes, pop_cons, species_info, center_nodes, center_cons, generation, jit_config):
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
    idx2specie = jnp.full((pop_size,), jnp.nan)  # I_INT means not assigned to any species

    # part 1: find new centers
    # the distance between each species' center and each genome in population
    s2p_distance = s2p_distance_func(center_nodes, center_cons, pop_nodes, pop_cons, jit_config)

    def find_new_centers(i, carry):
        i2s, cn, cc = carry
        # find new center
        idx = argmin_with_mask(s2p_distance[i], mask=jnp.isnan(i2s))

        # check species[i] exist or not
        # if not exist, set idx and i to I_INT, jax will not do array value assignment
        idx = jnp.where(~jnp.isnan(species_info[i, 0]), idx, I_INT)
        i = jnp.where(~jnp.isnan(species_info[i, 0]), i, I_INT)

        i2s = i2s.at[idx].set(species_info[i, 0])
        cn = cn.at[i].set(pop_nodes[idx])
        cc = cc.at[i].set(pop_cons[idx])
        return i2s, cn, cc

    idx2specie, center_nodes, center_cons = \
        jax.lax.fori_loop(0, species_size, find_new_centers, (idx2specie, center_nodes, center_cons))

    # part 2: assign members to each species
    def cond_func(carry):
        i, i2s, cn, cc, si, ck = carry  # si is short for species_info, ck is short for current key
        not_all_assigned = jnp.any(jnp.isnan(i2s))
        not_reach_species_upper_bounds = i < species_size
        return not_all_assigned & not_reach_species_upper_bounds

    def body_func(carry):
        i, i2s, cn, cc, si, ck = carry  # scn is short for spe_center_nodes, scc is short for spe_center_cons

        i2s, scn, scc, si, ck = jax.lax.cond(
            jnp.isnan(si[i, 0]),  # whether the current species is existing or not
            create_new_specie,  # if not existing, create a new specie
            update_exist_specie,  # if existing, update the specie
            (i, i2s, cn, cc, si, ck)
        )

        return i + 1, i2s, scn, scc, si, ck

    def create_new_specie(carry):
        i, i2s, cn, cc, si, ck = carry

        # pick the first one who has not been assigned to any species
        idx = fetch_first(jnp.isnan(i2s))

        # assign it to the new species
        # [key, best score, last update generation, members_count]
        si = si.at[i].set(jnp.array([ck, -jnp.inf, generation, 0]))
        i2s = i2s.at[idx].set(ck)

        # update center genomes
        cn = cn.at[i].set(pop_nodes[idx])
        cc = cc.at[i].set(pop_cons[idx])

        i2s = speciate_by_threshold((i, i2s, cn, cc, si))
        return i2s, cn, cc, si, ck + 1  # change to next new speciate key

    def update_exist_specie(carry):
        i, i2s, cn, cc, si, ck = carry
        i2s = speciate_by_threshold((i, i2s, cn, cc, si))
        return i2s, cn, cc, si, ck

    def speciate_by_threshold(carry):
        i, i2s, cn, cc, si = carry

        # distance between such center genome and ppo genomes
        o2p_distance = o2p_distance_func(cn[i], cc[i], pop_nodes, pop_cons, jit_config)
        close_enough_mask = o2p_distance < jit_config['compatibility_threshold']

        # when it is close enough, assign it to the species, remember not to update genome has already been assigned
        i2s = jnp.where(close_enough_mask & jnp.isnan(i2s), si[i, 0], i2s)
        return i2s

    species_keys = species_info[:, 0]
    current_new_key = jnp.max(jnp.where(jnp.isnan(species_keys), -jnp.inf, species_keys)) + 1

    # update idx2specie
    _, idx2specie, center_nodes, center_cons, species_info, _ = jax.lax.while_loop(
        cond_func,
        body_func,
        (0, idx2specie, center_nodes, center_cons, species_info, current_new_key)
    )

    # if there are still some pop genomes not assigned to any species, add them to the last genome
    # this condition seems to be only happened when the number of species is reached species upper bounds
    idx2specie = jnp.where(idx2specie == I_INT, species_info[-1, 0], idx2specie)

    # update members count
    def count_members(idx):
        key = species_info[idx, 0]
        count = jnp.sum(idx2specie == key)
        count = jnp.where(jnp.isnan(key), jnp.nan, count)
        return count

    species_member_counts = vmap(count_members)(jnp.arange(species_size))
    species_info = species_info.at[:, 3].set(species_member_counts)

    return idx2specie, center_nodes, center_cons, species_info


@jit
def argmin_with_mask(arr: Array, mask: Array) -> Array:
    masked_arr = jnp.where(mask, arr, jnp.inf)
    min_idx = jnp.argmin(masked_arr)
    return min_idx
