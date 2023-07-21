import jax.numpy as jnp
from algorithm.utils import unflatten_connections


def test_unflatten():
    nodes = jnp.array([
        [0, 0, 0, 0],
        [1, 1, 1, 1],
        [2, 2, 2, 2],
        [3, 3, 3, 3],
        [jnp.nan, jnp.nan, jnp.nan, jnp.nan]
    ])

    conns = jnp.array([
        [0, 1, True, 0.1, 0.11],
        [0, 2, False, 0.2, 0.22],
        [1, 2, True, 0.3, 0.33],
        [1, 3, False, 0.4, 0.44],
    ])

    res = unflatten_connections(nodes, conns)

    assert jnp.all(res[:, 0, 1] == jnp.array([True, 0.1, 0.11]))
    assert jnp.all(res[:, 0, 2] == jnp.array([False, 0.2, 0.22]))
    assert jnp.all(res[:, 1, 2] == jnp.array([True, 0.3, 0.33]))
    assert jnp.all(res[:, 1, 3] == jnp.array([False, 0.4, 0.44]))

    # Create a mask that excludes the indices we've already checked
    mask = jnp.ones(res.shape, dtype=bool)
    mask = mask.at[:, [0, 0, 1, 1], [1, 2, 2, 3]].set(False)

    # Ensure all other places are jnp.nan
    assert jnp.all(jnp.isnan(res[mask]))
