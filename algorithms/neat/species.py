from typing import List, Tuple, Dict, Union, Callable
from itertools import count

import jax
import numpy as np
from numpy.typing import NDArray


class Species(object):

    def __init__(self, key, generation):
        self.key = key
        self.created = generation
        self.last_improved = generation
        self.representative: Tuple[NDArray, NDArray] = (None, None)  # (nodes, connections)
        self.members: List[int] = []  # idx in pop_nodes, pop_connections,
        self.fitness = None
        self.member_fitnesses = None
        self.adjusted_fitness = None
        self.fitness_history: List[float] = []

    def update(self, representative, members):
        self.representative = representative
        self.members = members

    def get_fitnesses(self, fitnesses):
        return [fitnesses[m] for m in self.members]


class SpeciesController:
    """
    A class to control the species
    """

    def __init__(self, config):
        self.config = config
        self.compatibility_threshold = self.config.neat.species.compatibility_threshold
        self.species_elitism = self.config.neat.species.species_elitism
        self.pop_size = self.config.neat.population.pop_size
        self.max_stagnation = self.config.neat.species.max_stagnation
        self.min_species_size = self.config.neat.species.min_species_size
        self.genome_elitism = self.config.neat.species.genome_elitism
        self.survival_threshold = self.config.neat.species.survival_threshold

        self.species_idxer = count(0)
        self.species: Dict[int, Species] = {}  # species_id -> species

    def init_speciate(self, pop_nodes: NDArray, pop_connections: NDArray):
        """
        speciate for the first generation
        :param pop_connections:
        :param pop_nodes:
        :return:
        """
        pop_size = pop_nodes.shape[0]
        species_id = next(self.species_idxer)
        s = Species(species_id, 0)
        members = list(range(pop_size))
        s.update((pop_nodes[0], pop_connections[0]), members)
        self.species[species_id] = s

    def speciate(self, pop_nodes: NDArray, pop_connections: NDArray, generation: int,
                 o2o_distance: Callable, o2m_distance: Callable) -> None:
        """
        :param pop_nodes:
        :param pop_connections:
        :param generation: use to flag the created time of new species
        :param o2o_distance: distance function for one-to-one comparison
        :param o2m_distance: distance function for one-to-many comparison
        :return:
        """
        unspeciated = np.full((pop_nodes.shape[0],), True, dtype=bool)
        previous_species_list = list(self.species.keys())

        # Find the best representatives for each existing species.
        new_representatives = {}
        new_members = {}

        for sid, species in self.species.items():
            # calculate the distance between the representative and the population
            r_nodes, r_connections = species.representative
            distances = o2m_distance(r_nodes, r_connections, pop_nodes, pop_connections)
            distances = jax.device_get(distances)
            min_idx = find_min_with_mask(distances, unspeciated)  # find the min un-specified distance

            new_representatives[sid] = min_idx
            new_members[sid] = [min_idx]
            unspeciated[min_idx] = False

        # Partition population into species based on genetic similarity.

        # First, fast match the population to previous species
        if previous_species_list:  # exist previous species
            rid_list = [new_representatives[sid] for sid in previous_species_list]
            res_pop_distance = jax.device_get([
                o2m_distance(pop_nodes[rid], pop_connections[rid], pop_nodes, pop_connections)
                for rid in rid_list
            ])

            pop_res_distance = np.stack(res_pop_distance, axis=0).T
            for i in range(pop_res_distance.shape[0]):
                if not unspeciated[i]:
                    continue
                min_idx = np.argmin(pop_res_distance[i])
                min_val = pop_res_distance[i, min_idx]
                if min_val <= self.compatibility_threshold:
                    species_id = previous_species_list[min_idx]
                    new_members[species_id].append(i)
                    unspeciated[i] = False

        # Second, slowly match the lonely population to new-created species.
        # lonely genome is proved to be not compatible with any previous species, so they only need to be compared with
        # the new representatives.
        for i in range(pop_nodes.shape[0]):
            if not unspeciated[i]:
                continue
            unspeciated[i] = False
            if len(new_representatives) != 0:
                # the representatives of new species
                sid, rid = list(zip(*[(k, v) for k, v in new_representatives.items()]))
                distances = jax.device_get([
                    o2o_distance(pop_nodes[i], pop_connections[i], pop_nodes[r], pop_connections[r])
                    for r in rid
                ])
                distances = np.array(distances)
                min_idx = np.argmin(distances)
                min_val = distances[min_idx]
                if min_val <= self.compatibility_threshold:
                    species_id = sid[min_idx]
                    new_members[species_id].append(i)
                    continue
            # create a new species
            species_id = next(self.species_idxer)
            new_representatives[species_id] = i
            new_members[species_id] = [i]

        assert np.all(~unspeciated)

        # Update species collection based on new speciation.
        for sid, rid in new_representatives.items():
            s = self.species.get(sid)
            if s is None:
                s = Species(sid, generation)
                self.species[sid] = s

            members = new_members[sid]
            s.update((pop_nodes[rid], pop_connections[rid]), members)

    def update_species_fitnesses(self, fitnesses):
        """
        update the fitness of each species
        :param fitnesses:
        :return:
        """
        for sid, s in self.species.items():
            # TODO: here use mean to measure the fitness of a species, but it may be other functions
            s.member_fitnesses = s.get_fitnesses(fitnesses)
            # s.fitness = np.mean(s.member_fitnesses)
            s.fitness = np.max(s.member_fitnesses)
            s.fitness_history.append(s.fitness)
            s.adjusted_fitness = None

    def stagnation(self, generation):
        """
        code modified from neat-python!
        :param generation:
        :return: whether the species is stagnated
        """
        species_data = []
        for sid, s in self.species.items():
            if s.fitness_history:
                prev_fitness = max(s.fitness_history)
            else:
                prev_fitness = float('-inf')

            if prev_fitness is None or s.fitness > prev_fitness:
                s.last_improved = generation

            species_data.append((sid, s))

        # Sort in descending fitness order.
        species_data.sort(key=lambda x: x[1].fitness, reverse=True)

        result = []
        for idx, (sid, s) in enumerate(species_data):

            if idx < self.species_elitism:  # elitism species never stagnate!
                is_stagnant = False
            else:
                stagnant_time = generation - s.last_improved
                is_stagnant = stagnant_time > self.max_stagnation

            result.append((sid, s, is_stagnant))
        return result

    def reproduce(self, generation: int) -> List[Union[int, Tuple[int, int]]]:
        """
        code modified from neat-python!
        :param generation:
        :return: crossover_pair for next generation.
        # int -> idx in the pop_nodes, pop_connections of elitism
        # (int, int) -> the father and mother idx to be crossover
        """
        # Filter out stagnated species, collect the set of non-stagnated
        # species members, and compute their average adjusted fitness.
        # The average adjusted fitness scheme (normalized to the interval
        # [0, 1]) allows the use of negative fitness values without
        # interfering with the shared fitness scheme.
        all_fitnesses = []
        remaining_species = []
        for stag_sid, stag_s, stagnant in self.stagnation(generation):
            if not stagnant:
                all_fitnesses.extend(stag_s.member_fitnesses)
                remaining_species.append(stag_s)

        # No species left.
        if not remaining_species:
            self.species = {}
            return []

        # Compute each species' member size in the next generation.
        min_fitness = min(all_fitnesses)
        max_fitness = max(all_fitnesses)
        # Do not allow the fitness range to be zero, as we divide by it below.
        # TODO: The ``1.0`` below is rather arbitrary, and should be configurable.
        fitness_range = max(1.0, max_fitness - min_fitness)
        for afs in remaining_species:
            # Compute adjusted fitness.
            msf = afs.fitness
            af = (msf - min_fitness) / fitness_range  # make adjusted fitness in [0, 1]
            afs.adjusted_fitness = af
        adjusted_fitnesses = [s.adjusted_fitness for s in remaining_species]
        previous_sizes = [len(s.members) for s in remaining_species]
        min_species_size = max(self.min_species_size, self.genome_elitism)
        spawn_amounts = compute_spawn(adjusted_fitnesses, previous_sizes, self.pop_size, min_species_size)
        assert sum(spawn_amounts) == self.pop_size

        # generate new population and speciate
        self.species = {}
        # int -> idx in the pop_nodes, pop_connections of elitism
        # (int, int) -> the father and mother idx to be crossover

        crossover_pair: List[Union[int, Tuple[int, int]]] = []

        for spawn, s in zip(spawn_amounts, remaining_species):
            assert spawn >= self.genome_elitism

            # retain remain species to next generation
            old_members, fitnesses = s.members, s.member_fitnesses
            s.members = []
            self.species[s.key] = s

            # add elitism genomes to next generation
            sorted_members, sorted_fitnesses = sort_element_with_fitnesses(old_members, fitnesses)
            if self.genome_elitism > 0:
                for m in sorted_members[:self.genome_elitism]:
                    crossover_pair.append(m)
                    spawn -= 1

            if spawn <= 0:
                continue

            # add genome to be crossover to next generation
            repro_cutoff = int(np.ceil(self.survival_threshold * len(sorted_members)))
            repro_cutoff = max(repro_cutoff, 2)
            # only use good genomes to crossover
            sorted_members = sorted_members[:repro_cutoff]

            list_idx1, list_idx2 = np.random.choice(len(sorted_members), size=(2, spawn), replace=True)
            for c1, c2 in zip(list_idx1, list_idx2):
                idx1, fitness1 = sorted_members[c1], sorted_fitnesses[c1]
                idx2, fitness2 = sorted_members[c2], sorted_fitnesses[c2]
                if fitness1 >= fitness2:
                    crossover_pair.append((idx1, idx2))
                else:
                    crossover_pair.append((idx2, idx1))

        return crossover_pair


def compute_spawn(adjusted_fitness, previous_sizes, pop_size, min_species_size):
    """
    Code from neat-python, the only modification is to fix the population size for each generation.
    Compute the proper number of offspring per species (proportional to fitness).
    """
    af_sum = sum(adjusted_fitness)

    spawn_amounts = []
    for af, ps in zip(adjusted_fitness, previous_sizes):
        if af_sum > 0:
            s = max(min_species_size, af / af_sum * pop_size)
        else:
            s = min_species_size

        d = (s - ps) * 0.5
        c = int(round(d))
        spawn = ps
        if abs(c) > 0:
            spawn += c
        elif d > 0:
            spawn += 1
        elif d < 0:
            spawn -= 1

        spawn_amounts.append(spawn)

    # Normalize the spawn amounts so that the next generation is roughly
    # the population size requested by the user.
    total_spawn = sum(spawn_amounts)
    norm = pop_size / total_spawn
    spawn_amounts = [max(min_species_size, int(round(n * norm))) for n in spawn_amounts]

    # for batch parallelization, pop size must be a fixed value.
    total_amounts = sum(spawn_amounts)
    spawn_amounts[0] += pop_size - total_amounts
    assert sum(spawn_amounts) == pop_size, "Population size is not stable."

    return spawn_amounts


def find_min_with_mask(arr: NDArray, mask: NDArray) -> int:
    masked_arr = np.where(mask, arr, np.inf)
    min_idx = np.argmin(masked_arr)
    return min_idx


def sort_element_with_fitnesses(members: List[int], fitnesses: List[float]) \
        -> Tuple[List[int], List[float]]:
    combined = zip(members, fitnesses)
    sorted_combined = sorted(combined, key=lambda x: x[1], reverse=True)
    sorted_members = [item[0] for item in sorted_combined]
    sorted_fitnesses = [item[1] for item in sorted_combined]
    return sorted_members, sorted_fitnesses
