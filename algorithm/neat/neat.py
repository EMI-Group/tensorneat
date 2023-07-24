from typing import Type

import jax
from jax import numpy as jnp, Array, vmap
import numpy as np

from config import Config
from core import Algorithm, State, Gene, Genome
from .ga import crossover, create_mutate
from .species import SpeciesInfo, update_species, create_speciate


class NEAT(Algorithm):

    def __init__(self, config: Config, gene_type: Type[Gene]):
        self.config = config
        self.gene_type = gene_type

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
            NL=1 + len(self.gene_type.node_attrs),  # node length = (key) + attributes
            CL=3 + len(self.gene_type.conn_attrs),  # conn length = (in, out, key) + attributes
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

        state = self.gene_type.setup(self.config.gene, state)
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

        self.forward_func = self.gene_type.create_forward(state, self.config.gene)
        self.tell_func = self._create_tell()

        return jax.device_put(state)

    def ask(self, state: State):
        """require the population to be evaluated"""
        return state.pop_genomes

    def tell(self, state: State, fitness):
        """update the state of the algorithm"""
        return self.tell_func(state, fitness)

    def forward(self, inputs: Array, transformed: Array):
        """the forward function of a single forward transformation"""
        return self.forward_func(inputs, transformed)

    def forward_transform(self, state: State, genome: Genome):
        """create the forward transformation of a genome"""
        return self.gene_type.forward_transform(state, genome)

    def _initialize_genomes(self, state):
        o_nodes = np.full((state.N, state.NL), np.nan, dtype=np.float32)  # original nodes
        o_conns = np.full((state.C, state.CL), np.nan, dtype=np.float32)  # original connections

        input_idx = state.input_idx
        output_idx = state.output_idx
        new_node_key = max([*input_idx, *output_idx]) + 1

        o_nodes[input_idx, 0] = input_idx
        o_nodes[output_idx, 0] = output_idx
        o_nodes[new_node_key, 0] = new_node_key
        o_nodes[np.concatenate([input_idx, output_idx]), 1:] = self.gene_type.new_node_attrs(state)
        o_nodes[new_node_key, 1:] = self.gene_type.new_node_attrs(state)

        input_conns = np.c_[input_idx, np.full_like(input_idx, new_node_key)]
        o_conns[input_idx, 0:2] = input_conns  # in key, out key
        o_conns[input_idx, 2] = True  # enabled
        o_conns[input_idx, 3:] = self.gene_type.new_conn_attrs(state)

        output_conns = np.c_[np.full_like(output_idx, new_node_key), output_idx]
        o_conns[output_idx, 0:2] = output_conns  # in key, out key
        o_conns[output_idx, 2] = True  # enabled
        o_conns[output_idx, 3:] = self.gene_type.new_conn_attrs(state)

        # repeat origin genome for P times to create population
        pop_nodes = np.tile(o_nodes, (state.P, 1, 1))
        pop_conns = np.tile(o_conns, (state.P, 1, 1))

        return Genome(pop_nodes, pop_conns)

    def _create_tell(self):
        mutate = create_mutate(self.config.neat, self.gene_type)

        def create_next_generation(state, randkey, winner, loser, elite_mask):
            # prepare random keys
            pop_size = state.idx2species.shape[0]
            new_node_keys = jnp.arange(pop_size) + state.next_node_key

            k1, k2 = jax.random.split(randkey, 2)
            crossover_rand_keys = jax.random.split(k1, pop_size)
            mutate_rand_keys = jax.random.split(k2, pop_size)

            # batch crossover
            wpn, wpc = state.pop_genomes.nodes[winner], state.pop_genomes.conns[winner]
            lpn, lpc = state.pop_genomes.nodes[loser], state.pop_genomes.conns[loser]
            n_genomes = vmap(crossover)(crossover_rand_keys, Genome(wpn, wpc), Genome(lpn, lpc))

            # batch mutation
            mutate_func = vmap(mutate, in_axes=(None, 0, 0, 0))
            m_n_genomes = mutate_func(state, mutate_rand_keys, n_genomes, new_node_keys)  # mutate_new_pop_nodes

            # elitism don't mutate
            pop_nodes = jnp.where(elite_mask[:, None, None], n_genomes.nodes, m_n_genomes.nodes)
            pop_conns = jnp.where(elite_mask[:, None, None], n_genomes.conns, m_n_genomes.conns)

            # update next node key
            all_nodes_keys = pop_nodes[:, :, 0]
            max_node_key = jnp.max(jnp.where(jnp.isnan(all_nodes_keys), -jnp.inf, all_nodes_keys))
            next_node_key = max_node_key + 1

            return state.update(
                pop_genomes=Genome(pop_nodes, pop_conns),
                next_node_key=next_node_key,
            )

        speciate = create_speciate(self.gene_type)

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
