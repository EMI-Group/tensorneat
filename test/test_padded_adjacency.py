"""Tests for the bounded inbound adjacency used by DefaultGenome."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tensorneat.common import ACT, AGG, I_INF, check_cycles, check_cycles_flat
from tensorneat.genome import DefaultGenome, DefaultNode
from tensorneat.genome.operations import DefaultMutation
from tensorneat.genome.utils import (
    build_padded_inbound_adj,
    map_conn_endpoints,
    unflatten_conns,
)


def _genome(max_in_degree=None, **kwargs):
    defaults = {
        "num_inputs": 2,
        "num_outputs": 1,
        "max_nodes": 8,
        "max_conns": 16,
        "max_in_degree": max_in_degree,
    }
    defaults.update(kwargs)
    return DefaultGenome(**defaults)


def _initialized_pair(seed=0, max_in_degree=2):
    dense = _genome()
    padded = _genome(max_in_degree=max_in_degree)
    dense_state = dense.setup()
    padded_state = padded.setup()
    key = jax.random.PRNGKey(seed)
    nodes, conns = dense.initialize(dense_state, key)
    return dense, padded, dense_state, padded_state, nodes, conns


class TestConfiguration:
    def test_none_preserves_legacy_transform(self):
        genome = _genome()
        state = genome.setup()
        nodes, conns = genome.initialize(state, jax.random.PRNGKey(0))

        transformed = genome.transform(state, nodes, conns)

        assert genome.max_in_degree is None
        assert len(transformed) == 4
        assert transformed[-1].shape == (genome.max_nodes, genome.max_nodes)

    @pytest.mark.parametrize("value", [0, -1])
    def test_rejects_non_positive_cap(self, value):
        with pytest.raises(ValueError, match="greater than or equal to 1"):
            _genome(max_in_degree=value)

    def test_rejects_cap_larger_than_node_capacity(self):
        with pytest.raises(ValueError, match="exceeds max_nodes"):
            _genome(max_in_degree=9)

    def test_rejects_non_integer_cap(self):
        with pytest.raises(TypeError, match="integer or None"):
            _genome(max_in_degree=2.5)

    def test_rejects_initial_topology_over_cap(self):
        with pytest.raises(ValueError, match="initial topology"):
            _genome(max_in_degree=1)


class TestAdjacencyBuilder:
    def test_packs_slots_in_legacy_source_row_order(self):
        src_rows = jnp.array([2, 0, 1, I_INF], dtype=jnp.int32)
        dst_rows = jnp.array([3, 3, 3, I_INF], dtype=jnp.int32)
        valid = jnp.array([True, True, True, False])

        adj_conns, overflow = build_padded_inbound_adj(
            src_rows,
            dst_rows,
            valid,
            max_nodes=4,
            max_in_degree=3,
        )

        np.testing.assert_array_equal(
            np.asarray(adj_conns[3]), np.array([1, 2, 0])
        )
        assert not bool(overflow)

    def test_reports_overflow_and_keeps_exactly_the_cap(self):
        src_rows = jnp.array([0, 1, 2], dtype=jnp.int32)
        dst_rows = jnp.array([3, 3, 3], dtype=jnp.int32)
        valid = jnp.ones((3,), dtype=bool)

        adj_conns, overflow = build_padded_inbound_adj(
            src_rows,
            dst_rows,
            valid,
            max_nodes=4,
            max_in_degree=2,
        )

        assert bool(overflow)
        assert int(jnp.sum(adj_conns[3] != I_INF)) == 2

    def test_dangling_endpoints_are_invalid(self):
        nodes = jnp.array(
            [
                [0.0, 0.0],
                [1.0, 0.0],
                [jnp.nan, jnp.nan],
            ]
        )
        conns = jnp.array(
            [
                [0.0, 1.0, 1.0],
                [2.0, 1.0, 1.0],
                [jnp.nan, jnp.nan, jnp.nan],
            ]
        )

        src_rows, dst_rows, valid = map_conn_endpoints(nodes, conns)

        np.testing.assert_array_equal(np.asarray(valid), [True, False, False])
        assert int(src_rows[1]) == I_INF
        assert int(dst_rows[0]) == 1


class TestFlatGraphOperations:
    @pytest.mark.parametrize(
        ("from_idx", "to_idx", "expected"),
        [
            (3, 0, True),
            (1, 3, False),
            (2, 0, True),
            (1, 1, True),
        ],
    )
    def test_cycle_check_matches_dense_path(
        self, from_idx, to_idx, expected
    ):
        nodes = jnp.array([[0.0], [1.0], [2.0], [3.0], [jnp.nan]])
        conns = jnp.array(
            [
                [0.0, 2.0],
                [2.0, 3.0],
                [jnp.nan, jnp.nan],
                [jnp.nan, jnp.nan],
            ]
        )
        src_rows, dst_rows, valid = map_conn_endpoints(nodes, conns)
        dense_conns = unflatten_conns(nodes, conns) != I_INF

        dense_result = check_cycles(
            nodes, dense_conns, from_idx, to_idx
        )
        flat_result = jax.jit(check_cycles_flat)(
            nodes,
            src_rows,
            dst_rows,
            valid,
            from_idx,
            to_idx,
        )

        assert bool(flat_result) is expected
        assert bool(flat_result) == bool(dense_result)


class TestForwardCompatibility:
    @pytest.mark.parametrize("seed", [0, 1, 7])
    def test_initialized_network_matches_dense_path(self, seed):
        dense, padded, dense_state, padded_state, nodes, conns = (
            _initialized_pair(seed)
        )
        inputs = jnp.array([0.25, -0.5])

        dense_output = dense.forward(
            dense_state,
            dense.transform(dense_state, nodes, conns),
            inputs,
        )
        padded_output = padded.forward(
            padded_state,
            padded.transform(padded_state, nodes, conns),
            inputs,
        )

        np.testing.assert_array_equal(dense_output, padded_output)

    @pytest.mark.parametrize(
        "inputs",
        [
            jnp.array([0.25, -0.5]),
            jnp.array([1.0, 1.0]),
            jnp.array([-2.0, 0.125]),
        ],
    )
    def test_hidden_dag_matches_dense_path(self, inputs):
        dense, padded, dense_state, padded_state, nodes, _ = (
            _initialized_pair(3)
        )
        nodes = nodes.at[3].set(nodes[2].at[0].set(3.0))
        nodes = nodes.at[4].set(nodes[2].at[0].set(4.0))
        conns = jnp.full((16, 3), jnp.nan)
        conns = conns.at[:6].set(
            jnp.array(
                [
                    [4.0, 2.0, 0.75],
                    [1.0, 3.0, -0.5],
                    [3.0, 4.0, 1.25],
                    [1.0, 2.0, 0.1],
                    [0.0, 4.0, -0.25],
                    [0.0, 3.0, 0.8],
                ]
            )
        )

        dense_transformed = dense.transform(dense_state, nodes, conns)
        padded_transformed = padded.transform(padded_state, nodes, conns)
        dense_output = dense.forward(
            dense_state, dense_transformed, inputs
        )
        padded_output = padded.forward(
            padded_state, padded_transformed, inputs
        )

        np.testing.assert_array_equal(
            dense_transformed[0], padded_transformed[0]
        )
        np.testing.assert_array_equal(dense_output, padded_output)

    def test_preserves_floating_point_reduction_order(self):
        dense = _genome(
            num_inputs=3,
            max_nodes=4,
            max_conns=6,
        )
        padded = _genome(
            num_inputs=3,
            max_nodes=4,
            max_conns=6,
            max_in_degree=3,
        )
        dense_state = dense.setup()
        padded_state = padded.setup()
        nodes, _ = dense.initialize(dense_state, jax.random.PRNGKey(0))
        nodes = nodes.at[3, 1:].set(jnp.array([0.0, 1.0, 0.0, 0.0]))
        conns = jnp.full((6, 3), jnp.nan)
        conns = conns.at[:3].set(
            jnp.array(
                [
                    [1.0, 3.0, -1e20],
                    [2.0, 3.0, 1.0],
                    [0.0, 3.0, 1e20],
                ]
            )
        )
        inputs = jnp.ones((3,))

        dense_output = dense.forward(
            dense_state,
            dense.transform(dense_state, nodes, conns),
            inputs,
        )
        padded_output = padded.forward(
            padded_state,
            padded.transform(padded_state, nodes, conns),
            inputs,
        )

        np.testing.assert_array_equal(dense_output, padded_output)
        np.testing.assert_array_equal(dense_output, jnp.array([1.0]))

    def test_preserves_maxabs_tie_breaking(self):
        node_gene = DefaultNode(
            aggregation_options=AGG.maxabs,
            activation_options=ACT.identity,
        )
        dense = _genome(node_gene=node_gene)
        padded = _genome(max_in_degree=2, node_gene=node_gene)
        dense_state = dense.setup()
        padded_state = padded.setup()
        nodes, _ = dense.initialize(dense_state, jax.random.PRNGKey(0))
        nodes = nodes.at[2, 1:].set(jnp.array([0.0, 1.0, 0.0, 0.0]))
        conns = jnp.full((16, 3), jnp.nan)
        conns = conns.at[:2].set(
            jnp.array(
                [
                    [1.0, 2.0, 1.0],
                    [0.0, 2.0, 1.0],
                ]
            )
        )
        inputs = jnp.array([1.0, -1.0])

        dense_output = dense.forward(
            dense_state,
            dense.transform(dense_state, nodes, conns),
            inputs,
        )
        padded_output = padded.forward(
            padded_state,
            padded.transform(padded_state, nodes, conns),
            inputs,
        )

        np.testing.assert_array_equal(dense_output, padded_output)
        np.testing.assert_array_equal(dense_output, jnp.array([1.0]))

    def test_overflow_produces_nan_instead_of_truncated_evaluation(self):
        genome = _genome(
            max_in_degree=2,
            num_inputs=2,
            max_nodes=5,
            max_conns=8,
        )
        state = genome.setup()
        nodes, _ = genome.initialize(state, jax.random.PRNGKey(0))
        nodes = nodes.at[3].set(nodes[0].at[0].set(3.0))
        conns = jnp.full((8, 3), jnp.nan)
        conns = conns.at[:3].set(
            jnp.array(
                [
                    [0.0, 2.0, 1.0],
                    [1.0, 2.0, 1.0],
                    [3.0, 2.0, 1.0],
                ]
            )
        )

        transformed = genome.transform(state, nodes, conns)
        output = genome.forward(state, transformed, jnp.ones((2,)))

        assert bool(transformed[-1])
        assert jnp.all(jnp.isnan(output))

    def test_jit_and_vmap(self):
        genome = _genome(max_in_degree=2)
        state = genome.setup()
        keys = jax.random.split(jax.random.PRNGKey(0), 3)
        nodes, conns = jax.vmap(genome.initialize, in_axes=(None, 0))(
            state, keys
        )
        inputs = jnp.array([0.25, -0.5])

        def evaluate(nodes_, conns_):
            transformed = genome.transform(state, nodes_, conns_)
            return genome.forward(state, transformed, inputs)

        outputs = jax.jit(jax.vmap(evaluate))(nodes, conns)

        assert outputs.shape == (3, 1)
        assert jnp.all(jnp.isfinite(outputs))

    def test_gradients_are_finite(self):
        genome = _genome(max_in_degree=2)
        state = genome.setup()
        nodes, conns = genome.initialize(state, jax.random.PRNGKey(0))
        inputs = jnp.array([0.25, -0.5])

        def loss(nodes_, conns_):
            transformed = genome.transform(state, nodes_, conns_)
            output = genome.forward(state, transformed, inputs)
            return jnp.sum(output ** 2)

        grad_fn = jax.jit(jax.grad(loss, argnums=(0, 1)))
        grad_nodes, grad_conns = grad_fn(nodes, conns)

        assert jnp.all(jnp.isfinite(grad_nodes) | jnp.isnan(nodes))
        assert jnp.all(jnp.isfinite(grad_conns) | jnp.isnan(conns))


class TestMutationInvariant:
    def test_batched_mutation_matches_dense_path_below_cap(self):
        mutation = DefaultMutation(
            conn_add=1.0,
            conn_delete=0.0,
            node_add=0.0,
            node_delete=0.0,
        )
        dense = _genome(mutation=mutation)
        padded = _genome(max_in_degree=7, mutation=mutation)
        dense_state = dense.setup()
        padded_state = padded.setup()
        nodes, conns = dense.initialize(
            dense_state, jax.random.PRNGKey(0)
        )

        population = 8
        keys = jax.random.split(jax.random.PRNGKey(1), population)
        pop_nodes = jnp.repeat(nodes[None, ...], population, axis=0)
        pop_conns = jnp.repeat(conns[None, ...], population, axis=0)
        new_node_keys = jnp.arange(population, dtype=jnp.float32) + 100
        new_conn_markers = (
            jnp.arange(population * 3, dtype=jnp.float32)
            .reshape(population, 3)
            + 1000
        )

        def mutate_batch(genome, state):
            return jax.jit(
                jax.vmap(
                    genome.execute_mutation,
                    in_axes=(None, 0, 0, 0, 0, 0),
                )
            )(
                state,
                keys,
                pop_nodes,
                pop_conns,
                new_node_keys,
                new_conn_markers,
            )

        dense_nodes, dense_conns = mutate_batch(dense, dense_state)
        padded_nodes, padded_conns = mutate_batch(padded, padded_state)

        np.testing.assert_array_equal(dense_nodes, padded_nodes)
        np.testing.assert_array_equal(dense_conns, padded_conns)

    def test_add_connection_respects_full_destination(self):
        genome = _genome(
            max_in_degree=2,
            mutation=DefaultMutation(
                conn_add=1.0,
                conn_delete=0.0,
                node_add=0.0,
                node_delete=0.0,
            ),
        )
        state = genome.setup()
        nodes, conns = genome.initialize(state, jax.random.PRNGKey(0))
        initial_count = int(jnp.sum(~jnp.isnan(conns[:, 0])))

        for step in range(5):
            nodes, conns = genome.execute_mutation(
                state,
                jax.random.PRNGKey(step + 1),
                nodes,
                conns,
                jnp.array(100 + step, dtype=jnp.float32),
                jnp.arange(3, dtype=jnp.float32) + 1000 + 3 * step,
            )

        final_count = int(jnp.sum(~jnp.isnan(conns[:, 0])))
        assert final_count == initial_count
