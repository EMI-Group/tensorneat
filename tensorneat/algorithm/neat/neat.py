import jax, jax.numpy as jnp
from utils import State
from .. import BaseAlgorithm
from .species import *
from .ga import *
from .genome import *


class NEAT(BaseAlgorithm):
    def __init__(
        self,
        species: BaseSpecies,
        mutation: BaseMutation = DefaultMutation(),
        crossover: BaseCrossover = DefaultCrossover(),
    ):
        self.genome: BaseGenome = species.genome
        self.species = species
        self.mutation = mutation
        self.crossover = crossover

    def setup(self, state=State()):
        state = self.species.setup(state)
        state = self.mutation.setup(state)
        state = self.crossover.setup(state)
        state = state.register(
            generation=jnp.array(0.0),
            next_node_key=jnp.array(
                max(*self.genome.input_idx, *self.genome.output_idx) + 2,
                dtype=jnp.float32,
            ),
        )
        return state

    def ask(self, state: State):
        return state, self.species.ask(state.species)

    def tell(self, state: State, fitness):
        k1, k2, randkey = jax.random.split(state.randkey, 3)

        state = state.update(generation=state.generation + 1, randkey=randkey)

        state, winner, loser, elite_mask = self.species.update_species(
            state.species, fitness
        )
        state = self.create_next_generation(state, winner, loser, elite_mask)
        state = self.species.speciate(state.species)

        return state

    def transform(self, state, individual):
        """transform the genome into a neural network"""
        nodes, conns = individual
        return self.genome.transform(state, nodes, conns)

    def forward(self, state, inputs, transformed):
        return self.genome.forward(state, inputs, transformed)

    @property
    def num_inputs(self):
        return self.genome.num_inputs

    @property
    def num_outputs(self):
        return self.genome.num_outputs

    @property
    def pop_size(self):
        return self.species.pop_size

    def create_next_generation(self, state, winner, loser, elite_mask):
        # prepare random keys
        pop_size = self.species.pop_size
        new_node_keys = jnp.arange(pop_size) + state.next_node_key

        k1, k2, randkey = jax.random.split(state.randkey, 3)
        crossover_rand_keys = jax.random.split(k1, pop_size)
        mutate_rand_keys = jax.random.split(k2, pop_size)

        wpn, wpc = state.species.pop_nodes[winner], state.species.pop_conns[winner]
        lpn, lpc = state.species.pop_nodes[loser], state.species.pop_conns[loser]

        # batch crossover
        n_nodes, n_conns = jax.vmap(self.crossover, in_axes=(0, None, 0, 0, 0, 0))(
            crossover_rand_keys, self.genome, wpn, wpc, lpn, lpc
        )

        # batch mutation
        m_n_nodes, m_n_conns = jax.vmap(self.mutation, in_axes=(0, None, 0, 0, 0))(
            mutate_rand_keys, self.genome, n_nodes, n_conns, new_node_keys
        )

        # elitism don't mutate
        pop_nodes = jnp.where(elite_mask[:, None, None], n_nodes, m_n_nodes)
        pop_conns = jnp.where(elite_mask[:, None, None], n_conns, m_n_conns)

        # update next node key
        all_nodes_keys = pop_nodes[:, :, 0]
        max_node_key = jnp.max(
            jnp.where(jnp.isnan(all_nodes_keys), -jnp.inf, all_nodes_keys)
        )
        next_node_key = max_node_key + 1

        return state.update(
            randkey=randkey,
            pop_nodes=pop_nodes,
            pop_conns=pop_conns,
            next_node_key=next_node_key,
        )

    def member_count(self, state: State):
        return state, state.species.member_count

    def generation(self, state: State):
        # to analysis the algorithm
        return state, state.generation
