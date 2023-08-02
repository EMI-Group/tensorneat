from jax.tree_util import register_pytree_node_class
import numpy as np
import jax.numpy as jnp


@register_pytree_node_class
class SpeciesInfo:

    def __init__(self, species_keys, best_fitness, last_improved, member_count):
        self.species_keys = species_keys
        self.best_fitness = best_fitness
        self.last_improved = last_improved
        self.member_count = member_count

    @classmethod
    def initialize(cls, state):
        species_keys = np.full((state.S,), np.nan, dtype=np.float32)
        best_fitness = np.full((state.S,), np.nan, dtype=np.float32)
        last_improved = np.full((state.S,), np.nan, dtype=np.float32)
        member_count = np.full((state.S,), np.nan, dtype=np.float32)

        species_keys[0] = 0
        best_fitness[0] = -np.inf
        last_improved[0] = 0
        member_count[0] = state.P

        return cls(species_keys, best_fitness, last_improved, member_count)

    def __getitem__(self, i):
        return SpeciesInfo(self.species_keys[i], self.best_fitness[i], self.last_improved[i], self.member_count[i])

    def get(self, i):
        return self.species_keys[i], self.best_fitness[i], self.last_improved[i], self.member_count[i]

    def set(self, idx, value):
        species_keys = self.species_keys.at[idx].set(value[0])
        best_fitness = self.best_fitness.at[idx].set(value[1])
        last_improved = self.last_improved.at[idx].set(value[2])
        member_count = self.member_count.at[idx].set(value[3])
        return SpeciesInfo(species_keys, best_fitness, last_improved, member_count)

    def remove(self, idx):
        return self.set(idx, jnp.array([jnp.nan] * 4))

    def size(self):
        return self.species_keys.shape[0]

    def tree_flatten(self):
        children = self.species_keys, self.best_fitness, self.last_improved, self.member_count
        aux_data = None
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children)
