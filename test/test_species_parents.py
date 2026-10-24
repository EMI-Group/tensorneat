import jax
import jax.numpy as jnp

from tensorneat.algorithm.neat.species import SpeciesController
from tensorneat.common import State


def test_crossover_parents_stay_within_small_species():
    # Species 0 has a single member (individual 3); species 1 has everyone else.
    pop_size = 20
    controller = SpeciesController.__new__(SpeciesController)
    controller.pop_size = pop_size
    controller.species_size = 2
    controller.species_arange = jnp.arange(2)
    controller.survival_threshold = 0.1
    controller.genome_elitism = 2
    idx2species = jnp.where(jnp.arange(pop_size) == 3, 0.0, 1.0)
    species_state = State(species_keys=jnp.array([0.0, 1.0]), idx2species=idx2species)
    fitness = jnp.arange(pop_size, dtype=jnp.float32)
    spawn_number = jnp.array([3, pop_size - 3])
    winner, loser, elite_mask = controller._create_crossover_pair(
        species_state, jax.random.key(0), spawn_number, fitness
    )

    # Every parent of species 0's offspring is its only member, and only one elite is kept
    assert winner[:3].tolist() == [3, 3, 3]
    assert loser[:3].tolist() == [3, 3, 3]
    assert elite_mask[:3].tolist() == [True, False, False]

    # Species 1 has 19 members: parents come from it, and at least two of them survive
    assert bool((idx2species[winner[3:]] == 1).all())
    assert bool((idx2species[loser[3:]] == 1).all())
    assert len(set(winner[5:].tolist()) | set(loser[5:].tolist())) >= 2
