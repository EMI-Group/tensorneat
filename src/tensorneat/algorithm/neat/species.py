from typing import Callable

import jax
from jax import vmap, numpy as jnp
import numpy as np

from tensorneat.common import (
    State,
    StatefulBaseClass,
    rank_elements,
    argmin_with_mask,
    fetch_first,
)


class SpeciesController(StatefulBaseClass):
    def __init__(
        self,
        pop_size,
        species_size,
        max_stagnation,
        species_elitism,
        spawn_number_change_rate,
        genome_elitism,
        survival_threshold,
        min_species_size,
        compatibility_threshold,
        species_fitness_func,
        species_number_calculate_by,
    ):
        self.pop_size = pop_size
        self.species_size = species_size
        self.species_arange = np.arange(self.species_size)
        self.max_stagnation = max_stagnation
        self.species_elitism = species_elitism
        self.spawn_number_change_rate = spawn_number_change_rate
        self.genome_elitism = genome_elitism
        self.survival_threshold = survival_threshold
        self.min_species_size = min_species_size
        self.compatibility_threshold = compatibility_threshold
        self.species_fitness_func = species_fitness_func
        self.species_number_calculate_by = species_number_calculate_by

    def setup(self, state, first_nodes, first_conns):
        # the unique index (primary key) for each species
        species_keys = jnp.full((self.species_size,), jnp.nan)

        # the best fitness of each species
        best_fitness = jnp.full((self.species_size,), jnp.nan)

        # the last 1 that the species improved
        last_improved = jnp.full((self.species_size,), jnp.nan)

        # the number of members of each species
        member_count = jnp.full((self.species_size,), jnp.nan)

        # the species index of each individual
        idx2species = jnp.zeros(self.pop_size)

        # nodes for each center genome of each species
        center_nodes = jnp.full(
            (self.species_size, *first_nodes.shape),
            jnp.nan,
        )

        # connections for each center genome of each species
        center_conns = jnp.full(
            (self.species_size, *first_conns.shape),
            jnp.nan,
        )

        species_keys = species_keys.at[0].set(0)
        best_fitness = best_fitness.at[0].set(-jnp.inf)
        last_improved = last_improved.at[0].set(0)
        member_count = member_count.at[0].set(self.pop_size)
        center_nodes = center_nodes.at[0].set(first_nodes)
        center_conns = center_conns.at[0].set(first_conns)

        species_state = State(
            species_keys=species_keys,
            best_fitness=best_fitness,
            last_improved=last_improved,
            member_count=member_count,
            idx2species=idx2species,
            center_nodes=center_nodes,
            center_conns=center_conns,
            next_species_key=jnp.float32(1),  # 0 is reserved for the first species
        )

        return state.register(species=species_state)

    def update_species(self, state, fitness):
        species_state = state.species

        # update the fitness of each species
        species_fitness = self._update_species_fitness(species_state, fitness)

        # stagnation species
        species_state, species_fitness = self._stagnation(
            species_state, species_fitness, state.generation
        )

        # sort species_info by their fitness. (also push nan to the end)
        sort_indices = jnp.argsort(species_fitness)[::-1]  # fitness from high to low

        species_state = species_state.update(
            species_keys=species_state.species_keys[sort_indices],
            best_fitness=species_state.best_fitness[sort_indices],
            last_improved=species_state.last_improved[sort_indices],
            member_count=species_state.member_count[sort_indices],
            center_nodes=species_state.center_nodes[sort_indices],
            center_conns=species_state.center_conns[sort_indices],
        )

        # decide the number of members of each species by their fitness
        if self.species_number_calculate_by == "rank":
            spawn_number = self._cal_spawn_numbers_by_rank(species_state)
        elif self.species_number_calculate_by == "fitness":
            spawn_number = self._cal_spawn_numbers_by_fitness(species_state)
        else:
            raise ValueError("species_number_calculate_by must be 'rank' or 'fitness'")

        k1, k2 = jax.random.split(state.randkey)
        # crossover info
        winner, loser, elite_mask = self._create_crossover_pair(
            species_state, k1, spawn_number, fitness
        )

        return (
            state.update(randkey=k2, species=species_state),
            winner,
            loser,
            elite_mask,
        )

    def _update_species_fitness(self, species_state, fitness):
        """
        obtain the fitness of the species by the fitness of each individual.
        use max criterion.
        """

        def aux_func(idx):
            s_fitness = jnp.where(
                species_state.idx2species == species_state.species_keys[idx],
                fitness,
                -jnp.inf,
            )
            val = self.species_fitness_func(s_fitness)
            return val

        return vmap(aux_func)(self.species_arange)

    def _stagnation(self, species_state, species_fitness, generation):
        """
        stagnation species.
        those species whose fitness is not better than the best fitness of the species for a long time will be stagnation.
        elitism species never stagnation
        """

        def check_stagnation(idx):
            # determine whether the species stagnation

            # not better than the best fitness of the species
            # for a long time
            st = (species_fitness[idx] <= species_state.best_fitness[idx]) & (
                generation - species_state.last_improved[idx] > self.max_stagnation
            )

            # update last_improved and best_fitness
            # whether better than the best fitness of the species
            li, bf = jax.lax.cond(
                species_fitness[idx] > species_state.best_fitness[idx],
                lambda: (generation, species_fitness[idx]),  # update
                lambda: (
                    species_state.last_improved[idx],
                    species_state.best_fitness[idx],
                ),  # not update
            )

            return st, bf, li

        spe_st, best_fitness, last_improved = vmap(check_stagnation)(
            self.species_arange
        )

        # update species state
        species_state = species_state.update(
            best_fitness=best_fitness,
            last_improved=last_improved,
        )

        # elite species will not be stagnation
        species_rank = rank_elements(species_fitness)
        spe_st = jnp.where(
            species_rank < self.species_elitism, False, spe_st
        )  # elitism never stagnation

        # set stagnation species to nan
        def update_func(idx):
            return jax.lax.cond(
                spe_st[idx],
                lambda: (
                    jnp.nan,  # species_key
                    jnp.nan,  # best_fitness
                    jnp.nan,  # last_improved
                    jnp.nan,  # member_count
                    jnp.full_like(species_state.center_nodes[idx], jnp.nan),
                    jnp.full_like(species_state.center_conns[idx], jnp.nan),
                    -jnp.inf,  # species_fitness
                ),  # stagnation species
                lambda: (
                    species_state.species_keys[idx],
                    species_state.best_fitness[idx],
                    species_state.last_improved[idx],
                    species_state.member_count[idx],
                    species_state.center_nodes[idx],
                    species_state.center_conns[idx],
                    species_fitness[idx],
                ),  # not stagnation species
            )

        (
            species_keys,
            best_fitness,
            last_improved,
            member_count,
            center_nodes,
            center_conns,
            species_fitness,
        ) = vmap(update_func)(self.species_arange)

        return (
            species_state.update(
                species_keys=species_keys,
                best_fitness=best_fitness,
                last_improved=last_improved,
                member_count=member_count,
                center_nodes=center_nodes,
                center_conns=center_conns,
            ),
            species_fitness,
        )

    def _cal_spawn_numbers_by_rank(self, species_state):
        """
        decide the number of members of each species by their fitness rank.
        the species with higher fitness will have more members
        Linear ranking selection
            e.g. N = 3, P=10 -> probability = [0.5, 0.33, 0.17], spawn_number = [5, 3, 2]
        """

        species_keys = species_state.species_keys

        is_species_valid = ~jnp.isnan(species_keys)
        valid_species_num = jnp.sum(is_species_valid)
        denominator = (
            (valid_species_num + 1) * valid_species_num / 2
        )  # obtain 3 + 2 + 1 = 6

        # calculate the spawn number rate by the rank of each species
        rank_score = valid_species_num - self.species_arange  # obtain [3, 2, 1]
        spawn_number_rate = rank_score / denominator  # obtain [0.5, 0.33, 0.17]

        target_spawn_number = jnp.floor(
            spawn_number_rate * self.pop_size
        )  # calculate member

        # Avoid too much variation of numbers for a species
        previous_size = species_state.member_count
        spawn_number = (
            previous_size
            + (target_spawn_number - previous_size) * self.spawn_number_change_rate
        )

        # maintain min_species_size, this will not influence nan
        spawn_number = jnp.where(
            spawn_number < self.min_species_size, self.min_species_size, spawn_number
        )
        # convert to int, this will also make nan to 0
        spawn_number = spawn_number.astype(jnp.int32)

        # must control the sum of spawn_number to be equal to pop_size
        error = self.pop_size - jnp.sum(spawn_number)

        # add error to the first species to control the sum of spawn_number
        spawn_number = spawn_number.at[0].add(error)

        return spawn_number

    def _cal_spawn_numbers_by_fitness(self, species_state):
        """
        decide the number of members of each species by their fitness.
        the species with higher fitness will have more members
        """

        # the fitness of each species
        species_fitness = species_state.best_fitness

        # normalize the fitness before calculating the spawn number
        # consider that the fitness may be negative
        # in this way the species with the lowest fitness will have spawn_number = 0
        # 2025.1.31 updated, add +1 to avoid 0
        species_fitness = species_fitness - jnp.min(species_fitness) + 1

        # calculate the spawn number rate by the fitness of each species
        spawn_number_rate = species_fitness / jnp.sum(
            species_fitness, where=~jnp.isnan(species_fitness)
        )
        target_spawn_number = jnp.floor(
            spawn_number_rate * self.pop_size
        )  # calculate member

        # Avoid too much variation of numbers for a species
        previous_size = species_state.member_count
        spawn_number = (
            previous_size
            + (target_spawn_number - previous_size) * self.spawn_number_change_rate
        )
        # maintain min_species_size, this will not influence nan
        spawn_number = jnp.where(
            spawn_number < self.min_species_size, self.min_species_size, spawn_number
        )

        # convert to int, this will also make nan to 0
        spawn_number = spawn_number.astype(jnp.int32)

        # must control the sum of spawn_number to be equal to pop_size
        error = self.pop_size - jnp.sum(spawn_number)

        # add error to the first species to control the sum of spawn_number
        spawn_number = spawn_number.at[0].add(error)

        return spawn_number

    def _create_crossover_pair(self, species_state, randkey, spawn_number, fitness):
        s_idx = self.species_arange
        p_idx = jnp.arange(self.pop_size)

        def aux_func(key, idx):
            # choose parents from the in the same species
            # key -> randkey, idx -> the idx of current species

            members = species_state.idx2species == species_state.species_keys[idx]
            members_num = jnp.sum(members)

            members_fitness = jnp.where(members, fitness, -jnp.inf)
            sorted_member_indices = jnp.argsort(members_fitness)[::-1]

            survive_size = jnp.floor(self.survival_threshold * members_num).astype(
                jnp.int32
            )

            select_pro = (p_idx < survive_size) / survive_size
            fa, ma = jax.random.choice(
                key,
                sorted_member_indices,
                shape=(2, self.pop_size),
                replace=True,
                p=select_pro,
            )

            # elite
            fa = jnp.where(p_idx < self.genome_elitism, sorted_member_indices, fa)
            ma = jnp.where(p_idx < self.genome_elitism, sorted_member_indices, ma)
            elite = jnp.where(p_idx < self.genome_elitism, True, False)
            return fa, ma, elite

        # choose parents to crossover in each species
        # fas, mas, elites: (self.species_size, self.pop_size)
        # fas -> father indices, mas -> mother indices, elites -> whether elite or not
        fas, mas, elites = vmap(aux_func)(
            jax.random.split(randkey, self.species_size), s_idx
        )

        # merge choosen parents from each species into one array
        # winner, loser, elite_mask: (self.pop_size)
        # winner -> winner indices, loser -> loser indices, elite_mask -> whether elite or not
        spawn_number_cum = jnp.cumsum(spawn_number)

        def aux_func(idx):
            loc = jnp.argmax(idx < spawn_number_cum)

            # elite genomes are at the beginning of the species
            idx_in_species = jnp.where(loc > 0, idx - spawn_number_cum[loc - 1], idx)
            return (
                fas[loc, idx_in_species],
                mas[loc, idx_in_species],
                elites[loc, idx_in_species],
            )

        part1, part2, elite_mask = vmap(aux_func)(p_idx)

        is_part1_win = fitness[part1] >= fitness[part2]
        winner = jnp.where(is_part1_win, part1, part2)
        loser = jnp.where(is_part1_win, part2, part1)

        return winner, loser, elite_mask

    def speciate(self, state, genome_distance_func: Callable):
        # prepare distance functions
        o2p_distance_func = vmap(
            genome_distance_func, in_axes=(None, None, None, 0, 0)
        )  # one to population

        # idx to specie key
        idx2species = jnp.full(
            (self.pop_size,), jnp.nan
        )  # NaN means not assigned to any species

        # the distance between genomes to its center genomes
        o2c_distances = jnp.full((self.pop_size,), jnp.inf)

        # step 1: find new centers
        def cond_func(carry):
            # i, idx2species, center_nodes, center_conns, o2c_distances
            i, i2s, cns, ccs, o2c = carry

            return (i < self.species_size) & (
                ~jnp.isnan(state.species.species_keys[i])
            )  # current species is existing

        def body_func(carry):
            i, i2s, cns, ccs, o2c = carry

            distances = o2p_distance_func(
                state, cns[i], ccs[i], state.pop_nodes, state.pop_conns
            )

            # find the closest one
            closest_idx = argmin_with_mask(distances, mask=jnp.isnan(i2s))

            i2s = i2s.at[closest_idx].set(state.species.species_keys[i])
            cns = cns.at[i].set(state.pop_nodes[closest_idx])
            ccs = ccs.at[i].set(state.pop_conns[closest_idx])

            # the genome with closest_idx will become the new center, thus its distance to center is 0.
            o2c = o2c.at[closest_idx].set(0)

            return i + 1, i2s, cns, ccs, o2c

        _, idx2species, center_nodes, center_conns, o2c_distances = jax.lax.while_loop(
            cond_func,
            body_func,
            (
                0,
                idx2species,
                state.species.center_nodes,
                state.species.center_conns,
                o2c_distances,
            ),
        )

        state = state.update(
            species=state.species.update(
                idx2species=idx2species,
                center_nodes=center_nodes,
                center_conns=center_conns,
            ),
        )

        # part 2: assign members to each species
        def cond_func(carry):
            # i, idx2species, center_nodes, center_conns, species_keys, o2c_distances, next_species_key
            i, i2s, cns, ccs, sk, o2c, nsk = carry

            current_species_existed = ~jnp.isnan(sk[i])
            not_all_assigned = jnp.any(jnp.isnan(i2s))
            not_reach_species_upper_bounds = i < self.species_size
            return not_reach_species_upper_bounds & (
                current_species_existed | not_all_assigned
            )

        def body_func(carry):
            i, i2s, cns, ccs, sk, o2c, nsk = carry

            _, i2s, cns, ccs, sk, o2c, nsk = jax.lax.cond(
                jnp.isnan(sk[i]),  # whether the current species is existing or not
                create_new_species,  # if not existing, create a new specie
                update_exist_specie,  # if existing, update the specie
                (i, i2s, cns, ccs, sk, o2c, nsk),
            )

            return i + 1, i2s, cns, ccs, sk, o2c, nsk

        def create_new_species(carry):
            i, i2s, cns, ccs, sk, o2c, nsk = carry

            # pick the first one who has not been assigned to any species
            idx = fetch_first(jnp.isnan(i2s))

            # assign it to the new species
            # [key, best score, last update generation, member_count]
            sk = sk.at[i].set(nsk)  # nsk -> next species key
            i2s = i2s.at[idx].set(nsk)
            o2c = o2c.at[idx].set(0)

            # update center genomes
            cns = cns.at[i].set(state.pop_nodes[idx])
            ccs = ccs.at[i].set(state.pop_conns[idx])

            # find the members for the new species
            i2s, o2c = speciate_by_threshold(i, i2s, cns, ccs, sk, o2c)

            return i, i2s, cns, ccs, sk, o2c, nsk + 1  # change to next new speciate key

        def update_exist_specie(carry):
            i, i2s, cns, ccs, sk, o2c, nsk = carry

            i2s, o2c = speciate_by_threshold(i, i2s, cns, ccs, sk, o2c)

            # turn to next species
            return i + 1, i2s, cns, ccs, sk, o2c, nsk

        def speciate_by_threshold(i, i2s, cns, ccs, sk, o2c):
            # distance between such center genome and ppo genomes
            o2p_distance = o2p_distance_func(
                state, cns[i], ccs[i], state.pop_nodes, state.pop_conns
            )

            close_enough_mask = o2p_distance < self.compatibility_threshold
            # when a genome is not assigned or the distance between its current center is bigger than this center
            catchable_mask = jnp.isnan(i2s) | (o2p_distance < o2c)

            mask = close_enough_mask & catchable_mask

            # update species info
            i2s = jnp.where(mask, sk[i], i2s)

            # update distance between centers
            o2c = jnp.where(mask, o2p_distance, o2c)

            return i2s, o2c

        # update idx2species
        (
            _,
            idx2species,
            center_nodes,
            center_conns,
            species_keys,
            _,
            next_species_key,
        ) = jax.lax.while_loop(
            cond_func,
            body_func,
            (
                0,
                state.species.idx2species,
                center_nodes,
                center_conns,
                state.species.species_keys,
                o2c_distances,
                state.species.next_species_key,
            ),
        )

        # if there are still some pop genomes not assigned to any species, add them to the last genome
        # this condition can only happen when the number of species is reached species upper bounds
        idx2species = jnp.where(jnp.isnan(idx2species), species_keys[-1], idx2species)

        # complete info of species which is created in this generation
        new_created_mask = (~jnp.isnan(species_keys)) & jnp.isnan(
            state.species.best_fitness
        )
        best_fitness = jnp.where(new_created_mask, -jnp.inf, state.species.best_fitness)
        last_improved = jnp.where(
            new_created_mask, state.generation, state.species.last_improved
        )

        # update members count
        def count_members(idx):
            return jax.lax.cond(
                jnp.isnan(species_keys[idx]),  # if the species is not existing
                lambda: jnp.nan,  # nan
                lambda: jnp.sum(
                    idx2species == species_keys[idx], dtype=jnp.float32
                ),  # count members
            )

        member_count = vmap(count_members)(self.species_arange)

        species_state = state.species.update(
            species_keys=species_keys,
            best_fitness=best_fitness,
            last_improved=last_improved,
            member_count=member_count,
            idx2species=idx2species,
            center_nodes=center_nodes,
            center_conns=center_conns,
            next_species_key=next_species_key,
        )

        return state.update(
            species=species_state,
        )
