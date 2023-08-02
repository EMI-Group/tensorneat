from typing import Type

import jax
from jax import numpy as jnp
import numpy as np

from config import Config
from core import Algorithm, State, Gene, Genome
from .ga import create_next_generation
from .species import SpeciesInfo, update_species, speciate


class NEAT(Algorithm):

    def __init__(self, config: Config, gene_type: Type[Gene]):
        self.config = config
        self.gene = gene_type(config.gene)

        self.forward_func = None
        self.tell_func = None

    def setup(self, randkey, state: State = State()):
        """initialize the state of the algorithm"""

        input_idx = np.arange(self.config.neat.inputs)
        output_idx = np.arange(self.config.neat.inputs,
                               self.config.neat.inputs + self.config.neat.outputs)

        state = state.update(
            P=self.config.basic.pop_size,
            N=self.config.neat.maximum_nodes,
            C=self.config.neat.maximum_conns,
            S=self.config.neat.maximum_species,
            NL=1 + len(self.gene.node_attrs),  # node length = (key) + attributes
            CL=3 + len(self.gene.conn_attrs),  # conn length = (in, out, key) + attributes
            max_stagnation=self.config.neat.max_stagnation,
            species_elitism=self.config.neat.species_elitism,
            spawn_number_change_rate=self.config.neat.spawn_number_change_rate,
            genome_elitism=self.config.neat.genome_elitism,
            survival_threshold=self.config.neat.survival_threshold,
            compatibility_threshold=self.config.neat.compatibility_threshold,
            compatibility_disjoint=self.config.neat.compatibility_disjoint,
            compatibility_weight=self.config.neat.compatibility_weight,

            input_idx=input_idx,
            output_idx=output_idx,
        )

        state = self.gene.setup(state)
        pop_genomes = self._initialize_genomes(state)

        species_info = SpeciesInfo.initialize(state)
        idx2species = jnp.zeros(state.P, dtype=jnp.float32)

        center_nodes = jnp.full((state.S, state.N, state.NL), jnp.nan, dtype=jnp.float32)
        center_conns = jnp.full((state.S, state.C, state.CL), jnp.nan, dtype=jnp.float32)
        center_genomes = Genome(center_nodes, center_conns)
        center_genomes = center_genomes.set(0, pop_genomes[0])

        generation = 0
        next_node_key = max(*state.input_idx, *state.output_idx) + 2
        next_species_key = 1

        state = state.update(
            randkey=randkey,
            pop_genomes=pop_genomes,
            species_info=species_info,
            idx2species=idx2species,
            center_genomes=center_genomes,

            # avoid jax auto cast from int to float. that would cause re-compilation.
            generation=jnp.asarray(generation, dtype=jnp.int32),
            next_node_key=jnp.asarray(next_node_key, dtype=jnp.float32),
            next_species_key=jnp.asarray(next_species_key, dtype=jnp.float32),
        )

        return jax.device_put(state)

    def ask_algorithm(self, state: State):
        return state.pop_genomes

    def tell_algorithm(self, state: State, fitness):
        k1, k2, randkey = jax.random.split(state.randkey, 3)

        state = state.update(
            generation=state.generation + 1,
            randkey=randkey
        )

        state, winner, loser, elite_mask = update_species(state, k1, fitness)

        state = create_next_generation(self.config.neat, self.gene, state, k2, winner, loser, elite_mask)

        state = speciate(self.gene, state)

        return state

    def forward_transform(self, state: State, genome: Genome):
        return self.gene.forward_transform(state, genome)

    def forward(self, state: State, inputs, genome: Genome):
        return self.gene.forward(state, inputs, genome)

    def _initialize_genomes(self, state):
        o_nodes = np.full((state.N, state.NL), np.nan, dtype=np.float32)  # original nodes
        o_conns = np.full((state.C, state.CL), np.nan, dtype=np.float32)  # original connections

        input_idx = state.input_idx
        output_idx = state.output_idx
        new_node_key = max([*input_idx, *output_idx]) + 1

        o_nodes[input_idx, 0] = input_idx
        o_nodes[output_idx, 0] = output_idx
        o_nodes[new_node_key, 0] = new_node_key
        o_nodes[np.concatenate([input_idx, output_idx]), 1:] = self.gene.new_node_attrs(state)
        o_nodes[new_node_key, 1:] = self.gene.new_node_attrs(state)

        input_conns = np.c_[input_idx, np.full_like(input_idx, new_node_key)]
        o_conns[input_idx, 0:2] = input_conns  # in key, out key
        o_conns[input_idx, 2] = True  # enabled
        o_conns[input_idx, 3:] = self.gene.new_conn_attrs(state)

        output_conns = np.c_[np.full_like(output_idx, new_node_key), output_idx]
        o_conns[output_idx, 0:2] = output_conns  # in key, out key
        o_conns[output_idx, 2] = True  # enabled
        o_conns[output_idx, 3:] = self.gene.new_conn_attrs(state)

        # repeat origin genome for P times to create population
        pop_nodes = np.tile(o_nodes, (state.P, 1, 1))
        pop_conns = np.tile(o_conns, (state.P, 1, 1))

        return Genome(pop_nodes, pop_conns)
