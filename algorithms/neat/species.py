from typing import List, Tuple, Dict
from itertools import count

import jax
import numpy as np
from numpy.typing import NDArray
from .genome import distance


class Species(object):

    def __init__(self, key, generation):
        self.key = key
        self.created = generation
        self.last_improved = generation
        self.representative: Tuple[NDArray, NDArray] = (None, None)  # (nodes, connections)
        self.members: List[int] = []  # idx in pop_nodes, pop_connections
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
        self.max_stagnation = self.config.neat.species.max_stagnation

        self.species_idxer = count(0)
        self.species: Dict[int, Species] = {}  # species_id -> species
        self.genome_to_species: Dict[int, int] = {}

        self.o2m_distance_func = jax.vmap(distance, in_axes=(None, None, 0, 0))  # one to many
        # self.o2o_distance_func = np_distance  # one to one
        self.o2o_distance_func = distance

    def speciate(self, pop_nodes: NDArray, pop_connections: NDArray, generation: int) -> None:
        """
        :param pop_nodes:
        :param pop_connections:
        :param generation: use to flag the created time of new species
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
            distances = self.o2m_distance_func(r_nodes, r_connections, pop_nodes, pop_connections)
            distances = jax.device_get(distances)  # fetch the data from gpu
            min_idx = find_min_with_mask(distances, unspeciated)  # find the min un-specified distance

            new_representatives[sid] = min_idx
            new_members[sid] = [min_idx]
            unspeciated[min_idx] = False

        # Partition population into species based on genetic similarity.

        # First, fast match the population to previous species
        rid_list = [new_representatives[sid] for sid in previous_species_list]
        res_pop_distance = [
            jax.device_get(
                [
                    self.o2m_distance_func(pop_nodes[rid], pop_connections[rid], pop_nodes, pop_connections)
                    for rid in rid_list
                ]
            )
        ]
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
        new_species_list = []
        for i in range(pop_nodes.shape[0]):
            if not unspeciated[i]:
                continue
            unspeciated[i] = False
            if len(new_representatives) != 0:
                rid = [new_representatives[sid] for sid in new_representatives]  # the representatives of new species
                distances = [
                    self.o2o_distance_func(pop_nodes[i], pop_connections[i], pop_nodes[r], pop_connections[r])
                    for r in rid
                ]
                distances = np.array(distances)
                min_idx = np.argmin(distances)
                min_val = distances[min_idx]
                if min_val <= self.compatibility_threshold:
                    species_id = new_species_list[min_idx]
                    new_members[species_id].append(i)
                continue
            # create a new species
            species_id = next(self.species_idxer)
            new_species_list.append(species_id)
            new_representatives[species_id] = i
            new_members[species_id] = [i]

        assert np.all(~unspeciated)
        # Update species collection based on new speciation.
        self.genome_to_species = {}
        for sid, rid in new_representatives.items():
            s = self.species.get(sid)
            if s is None:
                s = Species(sid, generation)
                self.species[sid] = s

            members = new_members[sid]
            for gid in members:
                self.genome_to_species[gid] = sid

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
            s.fitness = np.mean(s.member_fitnesses)
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


def find_min_with_mask(arr: NDArray, mask: NDArray) -> int:
    masked_arr = np.where(mask, arr, np.inf)
    min_idx = np.argmin(masked_arr)
    return min_idx
