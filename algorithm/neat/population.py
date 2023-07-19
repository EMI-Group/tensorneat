from typing import Type

import jax
from jax import numpy as jnp, vmap

from .utils import rank_elements, fetch_first
from .genome import create_mutate, create_distance, crossover
from .gene import BaseGene


def create_tell(config, gene_type: Type[BaseGene]):
    mutate = create_mutate(config, gene_type)
    distance = create_distance(config, gene_type)

    def update_species(state, randkey, fitness):
        # update the fitness of each species
        species_fitness = update_species_fitness(state, fitness)

        # stagnation species
        state, species_fitness = stagnation(state, species_fitness)

        # sort species_info by their fitness. (push nan to the end)
        sort_indices = jnp.argsort(species_fitness)[::-1]

        state = state.update(
            species_info=state.species_info[sort_indices],
            center_nodes=state.center_nodes[sort_indices],
            center_conns=state.center_conns[sort_indices],
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
            species_key = state.species_info[idx, 0]
            s_fitness = jnp.where(state.idx2species == species_key, fitness, -jnp.inf)
            f = jnp.max(s_fitness)
            return f

        return vmap(aux_func)(jnp.arange(state.species_info.shape[0]))

    def stagnation(state, species_fitness):
        """
        stagnation species.
        those species whose fitness is not better than the best fitness of the species for a long time will be stagnation.
        elitism species never stagnation
        """

        def aux_func(idx):
            s_fitness = species_fitness[idx]
            species_key, best_score, last_update, members_count = state.species_info[idx]
            st = (s_fitness <= best_score) & (state.generation - last_update > state.max_stagnation)
            last_update = jnp.where(s_fitness > best_score, state.generation, last_update)
            best_score = jnp.where(s_fitness > best_score, s_fitness, best_score)
            # stagnation condition
            return st, jnp.array([species_key, best_score, last_update, members_count])

        spe_st, species_info = vmap(aux_func)(jnp.arange(species_fitness.shape[0]))

        # elite species will not be stagnation
        species_rank = rank_elements(species_fitness)
        spe_st = jnp.where(species_rank < state.species_elitism, False, spe_st)  # elitism never stagnation

        # set stagnation species to nan
        species_info = jnp.where(spe_st[:, None], jnp.nan, species_info)
        center_nodes = jnp.where(spe_st[:, None, None], jnp.nan, state.center_nodes)
        center_conns = jnp.where(spe_st[:, None, None], jnp.nan, state.center_conns)
        species_fitness = jnp.where(spe_st, -jnp.inf, species_fitness)

        state = state.update(
            species_info=species_info,
            center_nodes=center_nodes,
            center_conns=center_conns,
        )

        return state, species_fitness

    def cal_spawn_numbers(state):
        """
        decide the number of members of each species by their fitness rank.
        the species with higher fitness will have more members
        Linear ranking selection
            e.g. N = 3, P=10 -> probability = [0.5, 0.33, 0.17], spawn_number = [5, 3, 2]
        """

        is_species_valid = ~jnp.isnan(state.species_info[:, 0])
        valid_species_num = jnp.sum(is_species_valid)
        denominator = (valid_species_num + 1) * valid_species_num / 2  # obtain 3 + 2 + 1 = 6

        rank_score = valid_species_num - jnp.arange(state.species_info.shape[0])  # obtain [3, 2, 1]
        spawn_number_rate = rank_score / denominator  # obtain [0.5, 0.33, 0.17]
        spawn_number_rate = jnp.where(is_species_valid, spawn_number_rate, 0)  # set invalid species to 0

        target_spawn_number = jnp.floor(spawn_number_rate * state.P)  # calculate member

        # Avoid too much variation of numbers in a species
        previous_size = state.species_info[:, 3].astype(jnp.int32)
        spawn_number = previous_size + (target_spawn_number - previous_size) * state.spawn_number_change_rate
        # jax.debug.print("previous_size: {}, spawn_number: {}", previous_size, spawn_number)
        spawn_number = spawn_number.astype(jnp.int32)

        # spawn_number = target_spawn_number.astype(jnp.int32)

        # must control the sum of spawn_number to be equal to pop_size
        error = state.P - jnp.sum(spawn_number)
        spawn_number = spawn_number.at[0].add(
            error)  # add error to the first species to control the sum of spawn_number

        return spawn_number

    def create_crossover_pair(state, randkey, spawn_number, fitness):
        species_size = state.species_info.shape[0]
        pop_size = fitness.shape[0]
        s_idx = jnp.arange(species_size)
        p_idx = jnp.arange(pop_size)

        # def aux_func(key, idx):
        def aux_func(key, idx):
            members = state.idx2species == state.species_info[idx, 0]
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

    def create_next_generation(state, randkey, winner, loser, elite_mask):
        # prepare random keys
        pop_size = state.pop_nodes.shape[0]
        new_node_keys = jnp.arange(pop_size) + state.next_node_key

        k1, k2 = jax.random.split(randkey, 2)
        crossover_rand_keys = jax.random.split(k1, pop_size)
        mutate_rand_keys = jax.random.split(k2, pop_size)

        # batch crossover
        wpn, wpc = state.pop_nodes[winner], state.pop_conns[winner]  # winner pop nodes, winner pop connections
        lpn, lpc = state.pop_nodes[loser], state.pop_conns[loser]  # loser pop nodes, loser pop connections
        npn, npc = vmap(crossover)(crossover_rand_keys, wpn, wpc, lpn, lpc)  # new pop nodes, new pop connections

        # batch mutation
        mutate_func = vmap(mutate, in_axes=(None, 0, 0, 0, 0))
        m_npn, m_npc = mutate_func(state, mutate_rand_keys, npn, npc, new_node_keys)  # mutate_new_pop_nodes

        # elitism don't mutate
        pop_nodes = jnp.where(elite_mask[:, None, None], npn, m_npn)
        pop_conns = jnp.where(elite_mask[:, None, None], npc, m_npc)

        # update next node key
        all_nodes_keys = pop_nodes[:, :, 0]
        max_node_key = jnp.max(jnp.where(jnp.isnan(all_nodes_keys), -jnp.inf, all_nodes_keys))
        next_node_key = max_node_key + 1

        return state.update(
            pop_nodes=pop_nodes,
            pop_conns=pop_conns,
            next_node_key=next_node_key,
        )

    def speciate(state):
        pop_size, species_size = state.pop_nodes.shape[0], state.center_nodes.shape[0]

        # prepare distance functions
        o2p_distance_func = vmap(distance, in_axes=(None, None, None, 0, 0))  # one to population

        # idx to specie key
        idx2specie = jnp.full((pop_size,), jnp.nan)  # NaN means not assigned to any species

        # the distance between genomes to its center genomes
        o2c_distances = jnp.full((pop_size,), jnp.inf)

        # step 1: find new centers
        def cond_func(carry):
            i, i2s, cn, cc, o2c = carry
            species_key = state.species_info[i, 0]
            # jax.debug.print("{}, {}", i, species_key)
            return (i < species_size) & (~jnp.isnan(species_key))  # current species is existing

        def body_func(carry):
            i, i2s, cn, cc, o2c = carry
            distances = o2p_distance_func(state, cn[i], cc[i], state.pop_nodes, state.pop_conns)

            # find the closest one
            closest_idx = argmin_with_mask(distances, mask=jnp.isnan(i2s))
            # jax.debug.print("closest_idx: {}", closest_idx)

            i2s = i2s.at[closest_idx].set(state.species_info[i, 0])
            cn = cn.at[i].set(state.pop_nodes[closest_idx])
            cc = cc.at[i].set(state.pop_conns[closest_idx])

            # the genome with closest_idx will become the new center, thus its distance to center is 0.
            o2c = o2c.at[closest_idx].set(0)

            return i + 1, i2s, cn, cc, o2c

        _, idx2specie, center_nodes, center_conns, o2c_distances = \
            jax.lax.while_loop(cond_func, body_func,
                               (0, idx2specie, state.center_nodes, state.center_conns, o2c_distances))

        # part 2: assign members to each species
        def cond_func(carry):
            i, i2s, cn, cc, si, o2c, nsk = carry  # si is short for species_info, nsk is short for next_species_key
            current_species_existed = ~jnp.isnan(si[i, 0])
            not_all_assigned = jnp.any(jnp.isnan(i2s))
            not_reach_species_upper_bounds = i < species_size
            return not_reach_species_upper_bounds & (current_species_existed | not_all_assigned)

        def body_func(carry):
            i, i2s, cn, cc, si, o2c, nsk = carry  # scn is short for spe_center_nodes, scc is short for spe_center_conns

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
            si = si.at[i].set(jnp.array([nsk, -jnp.inf, state.generation, 0]))
            i2s = i2s.at[idx].set(nsk)
            o2c = o2c.at[idx].set(0)

            # update center genomes
            cn = cn.at[i].set(state.pop_nodes[idx])
            cc = cc.at[i].set(state.pop_conns[idx])

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
            o2p_distance = o2p_distance_func(state, cn[i], cc[i], state.pop_nodes, state.pop_conns)
            close_enough_mask = o2p_distance < state.compatibility_threshold

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
        _, idx2specie, center_nodes, center_conns, species_info, _, next_species_key = jax.lax.while_loop(
            cond_func,
            body_func,
            (0, idx2specie, center_nodes, center_conns, state.species_info, o2c_distances, state.next_species_key)
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

        return state.update(
            idx2species=idx2specie,
            center_nodes=center_nodes,
            center_conns=center_conns,
            species_info=species_info,
            next_species_key=next_species_key
        )

    def tell(state, fitness):
        """
        Main update function in NEAT.
        """

        k1, k2, randkey = jax.random.split(state.randkey, 3)

        state = state.update(
            generation=state.generation + 1,
            randkey=randkey
        )

        state, winner, loser, elite_mask = update_species(state, k1, fitness)

        state = create_next_generation(state, k2, winner, loser, elite_mask)

        state = speciate(state)

        return state

    return tell


def argmin_with_mask(arr, mask):
    masked_arr = jnp.where(mask, arr, jnp.inf)
    min_idx = jnp.argmin(masked_arr)
    return min_idx
