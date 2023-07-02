"""
Contains operations on the population: creating the next generation and population speciation.
The value tuple (P, N, C, S) is determined when the algorithm is initialized.
    P: population size
    N: maximum number of nodes in any genome
    C: maximum number of connections in any genome
    S: maximum number of species in NEAT

These arrays are used in the algorithm:
    fitness: Array[(P,), float], the fitness of each individual
    randkey: Array[2, uint], the random key
    pop_nodes: Array[(P, N, 5), float], nodes part of the population. [key, bias, response, act, agg]
    pop_cons: Array[(P, C, 4), float], connections part of the population. [in_node, out_node, weight, enabled]
    species_info: Array[(S, 4), float], the information of each species. [key, best_score, last_update, members_count]
    idx2species: Array[(P,), float], map the individual to its species keys
    center_nodes: Array[(S, N, 5), float], the center nodes of each species
    center_cons: Array[(S, C, 4), float], the center connections of each species
    generation: int, the current generation
    next_node_key: float, the next of the next node
    next_species_key: float, the next of the next species
    jit_config: Configer, the config used in jit-able functions
"""

# TODO: Complete python doc

import numpy as np
import jax
from jax import jit, vmap, Array, numpy as jnp

from .genome import initialize_genomes, distance, mutate, crossover, fetch_first, rank_elements


def initialize(config):
    """
    initialize the states of NEAT.
    """

    P = config['pop_size']
    N = config['maximum_nodes']
    C = config['maximum_connections']
    S = config['maximum_species']

    randkey = jax.random.PRNGKey(config['random_seed'])
    np.random.seed(config['random_seed'])
    pop_nodes, pop_cons = initialize_genomes(N, C, config)
    species_info = np.full((S, 4), np.nan, dtype=np.float32)
    species_info[0, :] = 0, -np.inf, 0, P
    idx2species = np.zeros(P, dtype=np.float32)
    center_nodes = np.full((S, N, 5), np.nan, dtype=np.float32)
    center_cons = np.full((S, C, 4), np.nan, dtype=np.float32)
    center_nodes[0, :, :] = pop_nodes[0, :, :]
    center_cons[0, :, :] = pop_cons[0, :, :]
    generation = np.asarray(0, dtype=np.int32)
    next_node_key = np.asarray(config['num_inputs'] + config['num_outputs'], dtype=np.float32)
    next_species_key = np.asarray(1, dtype=np.float32)

    return jax.device_put([
            randkey,
            pop_nodes,
            pop_cons,
            species_info,
            idx2species,
            center_nodes,
            center_cons,
            generation,
            next_node_key,
            next_species_key,
        ])

@jit
def tell(fitness,
         randkey,
         pop_nodes,
         pop_cons,
         species_info,
         idx2species,
         center_nodes,
         center_cons,
         generation,
         next_node_key,
         next_species_key,
         jit_config):
    """
    Main update function in NEAT.
    """
    generation += 1

    k1, k2, randkey = jax.random.split(randkey, 3)

    species_info, center_nodes, center_cons, winner, loser, elite_mask = \
        update_species(k1, fitness, species_info, idx2species, center_nodes,
                       center_cons, generation, jit_config)

    pop_nodes, pop_cons, next_node_key = create_next_generation(k2, pop_nodes, pop_cons, winner, loser,
                                                                elite_mask, next_node_key, jit_config)

    idx2species, center_nodes, center_cons, species_info, next_species_key = speciate(
        pop_nodes, pop_cons, species_info, center_nodes, center_cons, generation, next_species_key, jit_config)

    return randkey, pop_nodes, pop_cons, species_info, idx2species, center_nodes, center_cons, generation, next_node_key, next_species_key


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
    # jax.debug.print("spawn_number: {}", spawn_number)
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
    species_fitness = jnp.where(spe_st, -jnp.inf, species_fitness)

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

    target_spawn_number = jnp.floor(spawn_number_rate * jit_config['pop_size'])  # calculate member
    # jax.debug.print("denominator: {}, spawn_number_rate: {}, target_spawn_number: {}", denominator, spawn_number_rate, target_spawn_number)

    # Avoid too much variation of numbers in a species
    previous_size = species_info[:, 3].astype(jnp.int32)
    spawn_number = previous_size + (target_spawn_number - previous_size) * jit_config['spawn_number_move_rate']
    # jax.debug.print("previous_size: {}, spawn_number: {}", previous_size, spawn_number)
    spawn_number = spawn_number.astype(jnp.int32)

    # spawn_number = target_spawn_number.astype(jnp.int32)

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


def create_next_generation(rand_key, pop_nodes, pop_cons, winner, loser, elite_mask, next_node_key, jit_config):
    # prepare random keys
    pop_size = pop_nodes.shape[0]
    new_node_keys = jnp.arange(pop_size) + next_node_key

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

    # update next node key
    all_nodes_keys = pop_nodes[:, :, 0]
    max_node_key = jnp.max(jnp.where(jnp.isnan(all_nodes_keys), -jnp.inf, all_nodes_keys))
    next_node_key = max_node_key + 1

    return pop_nodes, pop_cons, next_node_key


def speciate(pop_nodes, pop_cons, species_info, center_nodes, center_cons, generation, next_species_key, jit_config):
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

    # idx to specie key
    idx2specie = jnp.full((pop_size,), jnp.nan)  # NaN means not assigned to any species

    # the distance between genomes to its center genomes
    o2c_distances = jnp.full((pop_size,), jnp.inf)

    # step 1: find new centers
    def cond_func(carry):
        i, i2s, cn, cc, o2c = carry
        species_key = species_info[i, 0]
        # jax.debug.print("{}, {}", i, species_key)
        return (i < species_size) & (~jnp.isnan(species_key))  # current species is existing

    def body_func(carry):
        i, i2s, cn, cc, o2c = carry
        distances = o2p_distance_func(cn[i], cc[i], pop_nodes, pop_cons, jit_config)

        # find the closest one
        closest_idx = argmin_with_mask(distances, mask=jnp.isnan(i2s))
        # jax.debug.print("closest_idx: {}", closest_idx)

        i2s = i2s.at[closest_idx].set(species_info[i, 0])
        cn = cn.at[i].set(pop_nodes[closest_idx])
        cc = cc.at[i].set(pop_cons[closest_idx])

        # the genome with closest_idx will become the new center, thus its distance to center is 0.
        o2c = o2c.at[closest_idx].set(0)

        return i + 1, i2s, cn, cc, o2c

    _, idx2specie, center_nodes, center_cons, o2c_distances = \
        jax.lax.while_loop(cond_func, body_func, (0, idx2specie, center_nodes, center_cons, o2c_distances))

    # jax.debug.print("species_info: \n{}", species_info)
    # jax.debug.print("idx2specie: \n{}", idx2specie)

    # part 2: assign members to each species
    def cond_func(carry):
        i, i2s, cn, cc, si, o2c, nsk = carry  # si is short for species_info, nsk is short for next_species_key
        # jax.debug.print("i:\n{}\ni2s:\n{}\nsi:\n{}", i, i2s, si)
        current_species_existed = ~jnp.isnan(si[i, 0])
        not_all_assigned = jnp.any(jnp.isnan(i2s))
        not_reach_species_upper_bounds = i < species_size
        return not_reach_species_upper_bounds & (current_species_existed | not_all_assigned)

    def body_func(carry):
        i, i2s, cn, cc, si, o2c, nsk = carry  # scn is short for spe_center_nodes, scc is short for spe_center_cons

        _, i2s, scn, scc, si, o2c, nsk = jax.lax.cond(
            jnp.isnan(si[i, 0]),  # whether the current species is existing or not
            create_new_species,  # if not existing, create a new specie
            update_exist_specie,  # if existing, update the specie
            (i, i2s, cn, cc, si, o2c, nsk)
        )

        return i + 1, i2s, scn, scc, si, o2c, nsk

    def create_new_species(carry):
        i, i2s, cn, cc, si, o2c, nsk = carry

        # pick the first one who has not been assigned to any species
        idx = fetch_first(jnp.isnan(i2s))

        # assign it to the new species
        # [key, best score, last update generation, members_count]
        si = si.at[i].set(jnp.array([nsk, -jnp.inf, generation, 0]))
        i2s = i2s.at[idx].set(nsk)
        o2c = o2c.at[idx].set(0)

        # update center genomes
        cn = cn.at[i].set(pop_nodes[idx])
        cc = cc.at[i].set(pop_cons[idx])

        i2s, o2c = speciate_by_threshold((i, i2s, cn, cc, si, o2c))

        # when a new species is created, it needs to be updated, thus do not change i
        return i + 1, i2s, cn, cc, si, o2c, nsk + 1  # change to next new speciate key

    def update_exist_specie(carry):
        i, i2s, cn, cc, si, o2c, nsk = carry
        i2s, o2c = speciate_by_threshold((i, i2s, cn, cc, si, o2c))

        # turn to next species
        return i + 1, i2s, cn, cc, si, o2c, nsk

    def speciate_by_threshold(carry):
        i, i2s, cn, cc, si, o2c = carry

        # distance between such center genome and ppo genomes
        o2p_distance = o2p_distance_func(cn[i], cc[i], pop_nodes, pop_cons, jit_config)
        close_enough_mask = o2p_distance < jit_config['compatibility_threshold']

        # when a genome is not assigned or the distance between its current center is bigger than this center
        cacheable_mask = jnp.isnan(i2s) | (o2p_distance < o2c)
        # jax.debug.print("{}", o2p_distance)
        mask = close_enough_mask & cacheable_mask

        # update species info
        i2s = jnp.where(mask, si[i, 0], i2s)

        # update distance between centers
        o2c = jnp.where(mask, o2p_distance, o2c)

        return i2s, o2c

    # update idx2specie
    _, idx2specie, center_nodes, center_cons, species_info, _, next_species_key = jax.lax.while_loop(
        cond_func,
        body_func,
        (0, idx2specie, center_nodes, center_cons, species_info, o2c_distances, next_species_key)
    )

    # if there are still some pop genomes not assigned to any species, add them to the last genome
    # this condition can only happen when the number of species is reached species upper bounds
    idx2specie = jnp.where(jnp.isnan(idx2specie), species_info[-1, 0], idx2specie)

    # update members count
    def count_members(idx):
        key = species_info[idx, 0]
        count = jnp.sum(idx2specie == key)
        count = jnp.where(jnp.isnan(key), jnp.nan, count)
        return count

    species_member_counts = vmap(count_members)(jnp.arange(species_size))
    species_info = species_info.at[:, 3].set(species_member_counts)

    return idx2specie, center_nodes, center_cons, species_info, next_species_key


def argmin_with_mask(arr: Array, mask: Array) -> Array:
    masked_arr = jnp.where(mask, arr, jnp.inf)
    min_idx = jnp.argmin(masked_arr)
    return min_idx
