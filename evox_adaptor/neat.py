import jax.numpy as jnp

import evox
from algorithms import neat
from configs import Configer


@evox.jit_class
class NEAT(evox.Algorithm):
    def __init__(self, config):
        self.config = config  # global config
        self.jit_config = Configer.create_jit_config(config)
        (
            self.randkey,
            self.pop_nodes,
            self.pop_cons,
            self.species_info,
            self.idx2species,
            self.center_nodes,
            self.center_cons,
            self.generation,
            self.next_node_key,
            self.next_species_key,
        ) = neat.initialize(config)
        super().__init__()

    def setup(self, key):
        return evox.State(
            randkey=self.randkey,
            pop_nodes=self.pop_nodes,
            pop_cons=self.pop_cons,
            species_info=self.species_info,
            idx2species=self.idx2species,
            center_nodes=self.center_nodes,
            center_cons=self.center_cons,
            generation=self.generation,
            next_node_key=self.next_node_key,
            next_species_key=self.next_species_key,
            jit_config=self.jit_config
        )

    def ask(self, state):
        flatten_pop_nodes = state.pop_nodes.flatten()
        flatten_pop_cons = state.pop_cons.flatten()
        pop = jnp.concatenate([flatten_pop_nodes, flatten_pop_cons])
        return pop, state

    def tell(self, state, fitness):

        # evox is a minimization framework, so we need to negate the fitness
        fitness = -fitness

        (
            randkey,
            pop_nodes,
            pop_cons,
            species_info,
            idx2species,
            center_nodes,
            center_cons,
            generation,
            next_node_key,
            next_species_key
        ) = neat.tell(
            fitness,
            state.randkey,
            state.pop_nodes,
            state.pop_cons,
            state.species_info,
            state.idx2species,
            state.center_nodes,
            state.center_cons,
            state.generation,
            state.next_node_key,
            state.next_species_key,
            state.jit_config
        )

        return evox.State(
            randkey=randkey,
            pop_nodes=pop_nodes,
            pop_cons=pop_cons,
            species_info=species_info,
            idx2species=idx2species,
            center_nodes=center_nodes,
            center_cons=center_cons,
            generation=generation,
            next_node_key=next_node_key,
            next_species_key=next_species_key,
            jit_config=state.jit_config
        )
