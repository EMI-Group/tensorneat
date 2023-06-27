from functools import partial

import jax
from jax import jit, numpy as jnp, vmap

from .genome.utils import rank_elements


@jit
def update_species(randkey, fitness, species_info, idx2species, center_nodes, center_cons, generation, jit_config):
    """
    args:
        randkey: random key
        fitness: Array[(pop_size,), float], the fitness of each individual
        species_keys: Array[(species_size, 3), float], the information of each species
            [species_key, best_score, last_update]
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

    jax.debug.print("{}, {}", fitness, winner)
    jax.debug.print("{}", fitness[winner])

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
        species_key, best_score, last_update = species_info[idx]
        # stagnation condition
        return (s_fitness <= best_score) & (generation - last_update > jit_config['max_stagnation'])

    st = vmap(aux_func)(jnp.arange(species_info.shape[0]))

    # elite species will not be stagnation
    species_rank = rank_elements(species_fitness)
    st = jnp.where(species_rank < jit_config['species_elitism'], False, st)  # elitism never stagnation

    # set stagnation species to nan
    species_info = jnp.where(st[:, None], jnp.nan, species_info)
    center_nodes = jnp.where(st[:, None, None], jnp.nan, center_nodes)
    center_cons = jnp.where(st[:, None, None], jnp.nan, center_cons)
    species_fitness = jnp.where(st, jnp.nan, species_fitness)

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

    def aux_func(key, idx):
        members = idx2species == species_info[idx, 0]
        members_num = jnp.sum(members)

        members_fitness = jnp.where(members, fitness, jnp.nan)
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
