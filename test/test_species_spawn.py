import jax.numpy as jnp

from tensorneat.algorithm.neat.species import SpeciesController
from tensorneat.common import State


def _controller(pop_size, species_size):
    controller = SpeciesController.__new__(SpeciesController)
    controller.pop_size = pop_size
    controller.species_size = species_size
    controller.spawn_number_change_rate = 1.0
    controller.min_species_size = 1
    controller.species_arange = jnp.arange(species_size)
    return controller


def test_spawn_numbers_by_fitness_ignore_empty_species_slots():
    # Three live species, sorted best first, and two empty (nan) slots
    controller = _controller(pop_size=100, species_size=5)
    species_state = State(
        species_keys=jnp.array([0.0, 1.0, 2.0, jnp.nan, jnp.nan]),
        best_fitness=jnp.array([10.0, 5.0, 1.0, jnp.nan, jnp.nan]),
        member_count=jnp.array([40.0, 30.0, 30.0, jnp.nan, jnp.nan]),
    )
    spawn_number = controller._cal_spawn_numbers_by_fitness(species_state)

    # Shifted fitness 10, 5, 1 -> floor(100 * [10, 5, 1] / 16) = [62, 31, 6], plus the rounding
    # error on the first species
    assert spawn_number.tolist() == [63, 31, 6, 0, 0]
