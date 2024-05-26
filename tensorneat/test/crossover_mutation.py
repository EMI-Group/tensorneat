import jax, jax.numpy as jnp
from utils import Act
from algorithm.neat import *
import numpy as np


def main():
    algorithm = NEAT(
        species=DefaultSpecies(
            genome=DefaultGenome(
                num_inputs=3,
                num_outputs=1,
                max_nodes=100,
                max_conns=100,
            ),
            pop_size=1000,
            species_size=10,
            compatibility_threshold=3.5,
        ),
        mutation=DefaultMutation(
            conn_add=0.4,
            conn_delete=0,
            node_add=0.9,
            node_delete=0,
        ),
    )

    state = algorithm.setup(jax.random.key(0))
    pop_nodes, pop_conns = algorithm.species.ask(state.species)

    batch_transform = jax.vmap(algorithm.genome.transform)
    batch_forward = jax.vmap(algorithm.forward, in_axes=(None, 0))

    for _ in range(50):
        winner, losser = jax.random.randint(state.randkey, (2, 1000), 0, 1000)
        elite_mask = jnp.zeros((1000,), dtype=jnp.bool_)
        elite_mask = elite_mask.at[:5].set(1)

        state = algorithm.create_next_generation(
            jax.random.key(0), state, winner, losser, elite_mask
        )
        pop_nodes, pop_conns = algorithm.species.ask(state.species)

        transforms = batch_transform(pop_nodes, pop_conns)
        outputs = batch_forward(jnp.array([1, 0, 1]), transforms)

        try:
            assert not jnp.any(jnp.isnan(outputs))
        except:
            print(_)


if __name__ == "__main__":
    main()
