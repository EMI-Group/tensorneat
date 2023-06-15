from typing import List, Tuple, Dict, Union, Callable
from itertools import count

import jax
import numpy as np
from numpy.typing import NDArray

from .genome.utils import I_INT


class Species(object):

    def __init__(self, key, generation):
        self.key = key
        self.created = generation
        self.last_improved = generation
        self.representative: Tuple[NDArray, NDArray] = (None, None)  # (center_nodes, center_connections)
        self.members: NDArray = None  # idx in pop_nodes, pop_connections,
        self.fitness = None
        self.member_fitnesses = None
        self.adjusted_fitness = None
        self.fitness_history: List[float] = []

    def update(self, representative, members):
        self.representative = representative
        self.members = members

    def get_fitnesses(self, fitnesses):
        return fitnesses[self.members]


class SpeciesController:
    """
    A class to control the species
    """

    def __init__(self, config):
        self.config = config

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
        members = np.array(list(range(pop_size)))
        s.update((pop_nodes[0], pop_connections[0]), members)
        self.species[species_id] = s

    def __update_species_fitnesses(self, fitnesses):
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

    def __stagnation(self, generation):
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

    def __reproduce(self, fitnesses: NDArray, generation: int) -> Tuple[NDArray, NDArray, NDArray]:
        """
        code modified from neat-python!
        :param fitnesses:
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

        min_fitness = np.inf
        max_fitness = -np.inf

        remaining_species = []
        for stag_sid, stag_s, stagnant in self.__stagnation(generation):
            if not stagnant:
                min_fitness = min(min_fitness, np.min(stag_s.member_fitnesses))
                max_fitness = max(max_fitness, np.max(stag_s.member_fitnesses))
                remaining_species.append(stag_s)

        # No species left.
        assert remaining_species

        # Compute each species' member size in the next generation.

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

        part1, part2, elite_mask = [], [], []

        for spawn, s in zip(spawn_amounts, remaining_species):
            assert spawn >= self.genome_elitism

            # retain remain species to next generation
            old_members, member_fitnesses = s.members, s.member_fitnesses
            s.members = []
            self.species[s.key] = s

            # add elitism genomes to next generation
            sorted_members, sorted_fitnesses = sort_element_with_fitnesses(old_members, member_fitnesses)
            if self.genome_elitism > 0:
                for m in sorted_members[:self.genome_elitism]:
                    part1.append(m)
                    part2.append(m)
                    elite_mask.append(True)
                    spawn -= 1

            if spawn <= 0:
                continue

            # add genome to be crossover to next generation
            repro_cutoff = int(np.ceil(self.survival_threshold * len(sorted_members)))
            repro_cutoff = max(repro_cutoff, 2)
            # only use good genomes to crossover
            sorted_members = sorted_members[:repro_cutoff]

            list_idx1, list_idx2 = np.random.choice(len(sorted_members), size=(2, spawn), replace=True)
            part1.extend(sorted_members[list_idx1])
            part2.extend(sorted_members[list_idx2])
            elite_mask.extend([False] * spawn)

        part1_fitness, part2_fitness = fitnesses[part1], fitnesses[part2]
        is_part1_win = part1_fitness >= part2_fitness
        winner_part = np.where(is_part1_win, part1, part2)
        loser_part = np.where(is_part1_win, part2, part1)

        return winner_part, loser_part, np.array(elite_mask)

    def tell(self, idx2specie, spe_center_nodes, spe_center_cons, species_keys, generation):
        for idx, key in enumerate(species_keys):
            if key == I_INT:
                continue

            members = np.where(idx2specie == key)[0]
            assert len(members) > 0
            if key not in self.species:
                s = Species(key, generation)
                self.species[key] = s

            self.species[key].update((spe_center_nodes[idx], spe_center_cons[idx]), members)

    def ask(self, fitnesses, generation, S, N, C):
        self.__update_species_fitnesses(fitnesses)
        winner_part, loser_part, elite_mask = self.__reproduce(fitnesses, generation)
        pre_spe_center_nodes = np.full((S, N, 5), np.nan)
        pre_spe_center_cons = np.full((S, C, 4), np.nan)
        species_keys = np.full((S,), I_INT)
        for idx, (key, specie) in enumerate(self.species.items()):
            pre_spe_center_nodes[idx] = specie.representative[0]
            pre_spe_center_cons[idx] = specie.representative[1]
            species_keys[idx] = key
        next_new_specie_key = max(self.species.keys()) + 1
        return winner_part, loser_part, elite_mask, pre_spe_center_nodes, \
            pre_spe_center_cons, species_keys, next_new_specie_key


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


def sort_element_with_fitnesses(members: NDArray, fitnesses: NDArray) \
        -> Tuple[NDArray, NDArray]:
    sorted_idx = np.argsort(fitnesses)[::-1]
    return members[sorted_idx], fitnesses[sorted_idx]
