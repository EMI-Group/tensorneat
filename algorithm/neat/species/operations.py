from typing import Type

import jax
from jax import numpy as jnp, vmap

from core import Gene, Genome
from utils import rank_elements, fetch_first
from .distance import create_distance


def update_species(state, randkey, fitness):
    # update the fitness of each species
    species_fitness = update_species_fitness(state, fitness)

    # stagnation species
    state, species_fitness = stagnation(state, species_fitness)

    # sort species_info by their fitness. (push nan to the end)
    sort_indices = jnp.argsort(species_fitness)[::-1]

    center_nodes = state.center_genomes.nodes[sort_indices]
    center_conns = state.center_genomes.conns[sort_indices]

    state = state.update(
        species_keys=state.species_keys[sort_indices],
        best_fitness=state.best_fitness[sort_indices],
        last_improved=state.last_improved[sort_indices],
        member_count=state.member_count[sort_indices],
        center_genomes=Genome(center_nodes, center_conns),
    )

    # decide the number of members of each species by their fitness
    spawn_number = cal_spawn_numbers(state)

    # crossover info
    winner, loser, elite_mask = create_crossover_pair(state, randkey, spawn_number, fitness)

    return state, winner, loser, elite_mask


def update_species_fitness(state, fitness):
    """
    obtain the fitness of the species by the fitness of each individual.
    use max criterion.
    """

    def aux_func(idx):
        s_fitness = jnp.where(state.idx2species == state.species_keys[idx], fitness, -jnp.inf)
        f = jnp.max(s_fitness)
        return f

    return vmap(aux_func)(jnp.arange(state.species_keys.shape[0]))


def stagnation(state, species_fitness):
    """
    stagnation species.
    those species whose fitness is not better than the best fitness of the species for a long time will be stagnation.
    elitism species never stagnation
    """

    def aux_func(idx):
        s_fitness = species_fitness[idx]
        sk, bf, li = state.species_keys[idx], state.best_fitness[idx], state.last_improved[idx]
        st = (s_fitness <= bf) & (state.generation - li > state.max_stagnation)
        li = jnp.where(s_fitness > bf, state.generation, li)
        bf = jnp.where(s_fitness > bf, s_fitness, bf)

        return st, sk, bf, li

    spe_st, species_keys, best_fitness, last_improved = vmap(aux_func)(jnp.arange(species_fitness.shape[0]))

    # elite species will not be stagnation
    species_rank = rank_elements(species_fitness)
    spe_st = jnp.where(species_rank < state.species_elitism, False, spe_st)  # elitism never stagnation

    # set stagnation species to nan
    species_keys = jnp.where(spe_st, jnp.nan, species_keys)
    best_fitness = jnp.where(spe_st, jnp.nan, best_fitness)
    last_improved = jnp.where(spe_st, jnp.nan, last_improved)
    member_count = jnp.where(spe_st, jnp.nan, state.member_count)
    species_fitness = jnp.where(spe_st, -jnp.inf, species_fitness)

    center_nodes = jnp.where(spe_st[:, None, None], jnp.nan, state.center_genomes.nodes)
    center_conns = jnp.where(spe_st[:, None, None], jnp.nan, state.center_genomes.conns)

    state = state.update(
        species_keys=species_keys,
        best_fitness=best_fitness,
        last_improved=last_improved,
        member_count=member_count,
        center_genomes=state.center_genomes.update(center_nodes, center_conns)
    )

    return state, species_fitness


def cal_spawn_numbers(state):
    """
    decide the number of members of each species by their fitness rank.
    the species with higher fitness will have more members
    Linear ranking selection
        e.g. N = 3, P=10 -> probability = [0.5, 0.33, 0.17], spawn_number = [5, 3, 2]
    """

    is_species_valid = ~jnp.isnan(state.species_keys)
    valid_species_num = jnp.sum(is_species_valid)
    denominator = (valid_species_num + 1) * valid_species_num / 2  # obtain 3 + 2 + 1 = 6

    rank_score = valid_species_num - jnp.arange(state.species_keys.shape[0])  # obtain [3, 2, 1]
    spawn_number_rate = rank_score / denominator  # obtain [0.5, 0.33, 0.17]
    spawn_number_rate = jnp.where(is_species_valid, spawn_number_rate, 0)  # set invalid species to 0

    target_spawn_number = jnp.floor(spawn_number_rate * state.P)  # calculate member

    # Avoid too much variation of numbers in a species
    previous_size = state.member_count
    spawn_number = previous_size + (target_spawn_number - previous_size) * state.spawn_number_change_rate
    # jax.debug.print("previous_size: {}, spawn_number: {}", previous_size, spawn_number)
    spawn_number = spawn_number.astype(jnp.int32)

    # must control the sum of spawn_number to be equal to pop_size
    error = state.P - jnp.sum(spawn_number)
    spawn_number = spawn_number.at[0].add(error)  # add error to the first species to control the sum of spawn_number

    return spawn_number


def create_crossover_pair(state, randkey, spawn_number, fitness):
    species_size = state.species_keys.shape[0]
    pop_size = fitness.shape[0]
    s_idx = jnp.arange(species_size)
    p_idx = jnp.arange(pop_size)

    # def aux_func(key, idx):
    def aux_func(key, idx):
        members = state.idx2species == state.species_keys[idx]
        members_num = jnp.sum(members)

        members_fitness = jnp.where(members, fitness, -jnp.inf)
        sorted_member_indices = jnp.argsort(members_fitness)[::-1]

        elite_size = state.genome_elitism
        survive_size = jnp.floor(state.survival_threshold * members_num).astype(jnp.int32)

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


def create_speciate(gene_type: Type[Gene]):
    distance = create_distance(gene_type)

    def speciate(state):
        pop_size, species_size = state.idx2species.shape[0], state.species_keys.shape[0]

        # prepare distance functions
        o2p_distance_func = vmap(distance, in_axes=(None, None, 0))  # one to population

        # idx to specie key
        idx2species = jnp.full((pop_size,), jnp.nan)  # NaN means not assigned to any species

        # the distance between genomes to its center genomes
        o2c_distances = jnp.full((pop_size,), jnp.inf)

        # step 1: find new centers
        def cond_func(carry):
            i, i2s, cgs, o2c = carry

            return (i < species_size) & (~jnp.isnan(state.species_keys[i]))  # current species is existing

        def body_func(carry):
            i, i2s, cgs, o2c = carry

            distances = o2p_distance_func(state, Genome(cgs.nodes[i], cgs.conns[i]), state.pop_genomes)

            # find the closest one
            closest_idx = argmin_with_mask(distances, mask=jnp.isnan(i2s))
            # jax.debug.print("closest_idx: {}", closest_idx)

            i2s = i2s.at[closest_idx].set(state.species_keys[i])
            cn = cgs.nodes.at[i].set(state.pop_genomes.nodes[closest_idx])
            cc = cgs.conns.at[i].set(state.pop_genomes.conns[closest_idx])

            # the genome with closest_idx will become the new center, thus its distance to center is 0.
            o2c = o2c.at[closest_idx].set(0)

            return i + 1, i2s, Genome(cn, cc), o2c

        _, idx2species, center_genomes, o2c_distances = \
            jax.lax.while_loop(cond_func, body_func, (0, idx2species, state.center_genomes, o2c_distances))

        state = state.update(
            idx2species=idx2species,
            center_genomes=center_genomes,
        )

        # part 2: assign members to each species
        def cond_func(carry):
            i, i2s, cgs, sk, o2c, nsk = carry

            current_species_existed = ~jnp.isnan(sk[i])
            not_all_assigned = jnp.any(jnp.isnan(i2s))
            not_reach_species_upper_bounds = i < species_size
            return not_reach_species_upper_bounds & (current_species_existed | not_all_assigned)

        def body_func(carry):
            i, i2s, cgs, sk, o2c, nsk = carry

            _, i2s, cgs, sk, o2c, nsk = jax.lax.cond(
                jnp.isnan(sk[i]),  # whether the current species is existing or not
                create_new_species,  # if not existing, create a new specie
                update_exist_specie,  # if existing, update the specie
                (i, i2s, cgs, sk, o2c, nsk)
            )

            return i + 1, i2s, cgs, sk, o2c, nsk

        def create_new_species(carry):
            i, i2s, cgs, sk, o2c, nsk = carry

            # pick the first one who has not been assigned to any species
            idx = fetch_first(jnp.isnan(i2s))

            # assign it to the new species
            # [key, best score, last update generation, members_count]
            sk = sk.at[i].set(nsk)
            i2s = i2s.at[idx].set(nsk)
            o2c = o2c.at[idx].set(0)

            # update center genomes
            cn = cgs.nodes.at[i].set(state.pop_genomes.nodes[idx])
            cc = cgs.conns.at[i].set(state.pop_genomes.conns[idx])
            cgs = Genome(cn, cc)

            i2s, o2c = speciate_by_threshold(i, i2s, cgs, sk, o2c)

            # when a new species is created, it needs to be updated, thus do not change i
            return i + 1, i2s, cgs, sk, o2c, nsk + 1  # change to next new speciate key

        def update_exist_specie(carry):
            i, i2s, cgs, sk, o2c, nsk = carry

            i2s, o2c = speciate_by_threshold(i, i2s, cgs, sk, o2c)

            # turn to next species
            return i + 1, i2s, cgs, sk, o2c, nsk

        def speciate_by_threshold(i, i2s, cgs, sk, o2c):
            # distance between such center genome and ppo genomes

            center = Genome(cgs.nodes[i], cgs.conns[i])
            o2p_distance = o2p_distance_func(state, center, state.pop_genomes)
            close_enough_mask = o2p_distance < state.compatibility_threshold

            # when a genome is not assigned or the distance between its current center is bigger than this center
            cacheable_mask = jnp.isnan(i2s) | (o2p_distance < o2c)
            # jax.debug.print("{}", o2p_distance)
            mask = close_enough_mask & cacheable_mask

            # update species info
            i2s = jnp.where(mask, sk[i], i2s)

            # update distance between centers
            o2c = jnp.where(mask, o2p_distance, o2c)

            return i2s, o2c

        # update idx2species
        _, idx2species, center_genomes, species_keys, _, next_species_key = jax.lax.while_loop(
            cond_func,
            body_func,
            (0, state.idx2species, state.center_genomes, state.species_keys, o2c_distances, state.next_species_key)
        )

        # if there are still some pop genomes not assigned to any species, add them to the last genome
        # this condition can only happen when the number of species is reached species upper bounds
        idx2species = jnp.where(jnp.isnan(idx2species), species_keys[-1], idx2species)

        # complete info of species which is created in this generation
        new_created_mask = (~jnp.isnan(species_keys)) & jnp.isnan(state.best_fitness)
        best_fitness = jnp.where(new_created_mask, -jnp.inf, state.best_fitness)
        last_improved = jnp.where(new_created_mask, state.generation, state.last_improved)

        # update members count
        def count_members(idx):
            key = species_keys[idx]
            count = jnp.sum(idx2species == key)
            count = jnp.where(jnp.isnan(key), jnp.nan, count)
            return count

        member_count = vmap(count_members)(jnp.arange(species_size))

        return state.update(
            species_keys=species_keys,
            best_fitness=best_fitness,
            last_improved=last_improved,
            members_count=member_count,
            idx2species=idx2species,
            center_genomes=center_genomes,
            next_species_key=next_species_key
        )

    return speciate


def argmin_with_mask(arr, mask):
    masked_arr = jnp.where(mask, arr, jnp.inf)
    min_idx = jnp.argmin(masked_arr)
    return min_idx
