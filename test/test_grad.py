"""Tests for gradient correctness of DefaultGenome and RecurrentGenome forward."""

import jax
import jax.numpy as jnp
import pytest

from tensorneat import genome
from tensorneat.common import ACT, AGG


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_genome_and_individual(genome_obj, seed=0):
    """Setup genome, initialize one individual, return (state, nodes, conns)."""
    state = genome_obj.setup()
    key = jax.random.PRNGKey(seed)
    nodes, conns = genome_obj.initialize(state, key)
    return state, nodes, conns


def _forward_fn(genome_obj, state, nodes, conns, inputs):
    """Full forward: transform then forward."""
    transformed = genome_obj.transform(state, nodes, conns)
    return genome_obj.forward(state, transformed, inputs)


def _is_connected(genome_obj, state, nodes, conns, inputs):
    """Check whether the network output contains NaN (disconnected)."""
    out = _forward_fn(genome_obj, state, nodes, conns, inputs)
    return jnp.all(jnp.isfinite(out))


# ---------------------------------------------------------------------------
# 1. Basic: jax.grad produces finite gradients for connected networks
# ---------------------------------------------------------------------------

class TestGradFinite:

    @pytest.fixture(params=[
        genome.DefaultGenome(num_inputs=3, num_outputs=1, max_nodes=7, max_conns=20,
                             output_transform=ACT.sigmoid),
        genome.DefaultGenome(num_inputs=2, num_outputs=2, max_nodes=6, max_conns=15,
                             output_transform=ACT.tanh),
    ], ids=["3in_1out_sigmoid", "2in_2out_tanh"])
    def connected_setup(self, request):
        g = request.param
        state, nodes, conns = _make_genome_and_individual(g, seed=42)
        inputs = jnp.ones(g.num_inputs) * 0.5
        assert _is_connected(g, state, nodes, conns, inputs), \
            "Initial network should be connected"
        return g, state, nodes, conns, inputs

    def test_grad_wrt_nodes_finite(self, connected_setup):
        g, state, nodes, conns, inputs = connected_setup

        def loss_fn(n):
            transformed = g.transform(state, n, conns)
            out = g.forward(state, transformed, inputs)
            return jnp.sum(out ** 2)

        grad_nodes = jax.grad(loss_fn)(nodes)
        assert jnp.all(jnp.isfinite(grad_nodes) | jnp.isnan(nodes)), \
            "Gradient w.r.t. nodes should be finite for valid entries"

    def test_grad_wrt_conns_finite(self, connected_setup):
        g, state, nodes, conns, inputs = connected_setup

        def loss_fn(c):
            transformed = g.transform(state, nodes, c)
            out = g.forward(state, transformed, inputs)
            return jnp.sum(out ** 2)

        grad_conns = jax.grad(loss_fn)(conns)
        assert jnp.all(jnp.isfinite(grad_conns) | jnp.isnan(conns)), \
            "Gradient w.r.t. conns should be finite for valid entries"

    def test_grad_wrt_both_finite(self, connected_setup):
        g, state, nodes, conns, inputs = connected_setup

        def loss_fn(n, c):
            transformed = g.transform(state, n, c)
            out = g.forward(state, transformed, inputs)
            return jnp.sum(out ** 2)

        grad_n, grad_c = jax.grad(loss_fn, argnums=(0, 1))(nodes, conns)
        valid_n = jnp.isfinite(grad_n) | jnp.isnan(nodes)
        valid_c = jnp.isfinite(grad_c) | jnp.isnan(conns)
        assert jnp.all(valid_n) and jnp.all(valid_c)


# ---------------------------------------------------------------------------
# 2. jit(grad) works
# ---------------------------------------------------------------------------

class TestJitGrad:

    def test_jit_grad_default_genome(self):
        g = genome.DefaultGenome(num_inputs=3, num_outputs=1, max_nodes=7, max_conns=20,
                                 output_transform=ACT.sigmoid)
        state, nodes, conns = _make_genome_and_individual(g, seed=0)
        inputs = jnp.array([1.0, 0.0, 1.0])

        def loss_fn(n, c):
            transformed = g.transform(state, n, c)
            out = g.forward(state, transformed, inputs)
            return jnp.sum(out ** 2)

        grad_fn = jax.jit(jax.grad(loss_fn, argnums=(0, 1)))
        grad_n, grad_c = grad_fn(nodes, conns)

        assert jnp.all(jnp.isfinite(grad_n) | jnp.isnan(nodes))
        assert jnp.all(jnp.isfinite(grad_c) | jnp.isnan(conns))


# ---------------------------------------------------------------------------
# 3. Gradient descent actually reduces loss
# ---------------------------------------------------------------------------

class TestGradDescentReducesLoss:

    def test_loss_decreases_default_genome(self):
        g = genome.DefaultGenome(num_inputs=3, num_outputs=1, max_nodes=7, max_conns=20,
                                 output_transform=ACT.sigmoid)
        state, nodes, conns = _make_genome_and_individual(g, seed=42)
        inputs = jnp.array([1.0, 0.0, 1.0])
        target = jnp.array([0.8])

        def loss_fn(n, c):
            transformed = g.transform(state, n, c)
            out = g.forward(state, transformed, inputs)
            return jnp.mean((out - target) ** 2)

        initial_loss = loss_fn(nodes, conns)
        grad_fn = jax.jit(jax.grad(loss_fn, argnums=(0, 1)))

        n, c = nodes, conns
        for _ in range(50):
            gn, gc = grad_fn(n, c)
            gn = jnp.where(jnp.isnan(nodes), 0.0, gn)
            gc = jnp.where(jnp.isnan(conns), 0.0, gc)
            n = n - 0.1 * gn
            c = c - 0.1 * gc

        final_loss = loss_fn(n, c)
        assert final_loss < initial_loss, \
            f"Loss should decrease after gradient steps: {initial_loss} -> {final_loss}"


# ---------------------------------------------------------------------------
# 4. Batch grad via vmap across multiple networks
# ---------------------------------------------------------------------------

class TestBatchGrad:

    def test_vmap_grad_multiple_networks(self):
        g = genome.DefaultGenome(num_inputs=2, num_outputs=1, max_nodes=5, max_conns=10,
                                 output_transform=ACT.sigmoid)
        state = g.setup()

        keys = jax.random.split(jax.random.PRNGKey(0), 4)
        batch_nodes = []
        batch_conns = []
        for k in keys:
            n, c = g.initialize(state, k)
            batch_nodes.append(n)
            batch_conns.append(c)
        batch_nodes = jnp.stack(batch_nodes)
        batch_conns = jnp.stack(batch_conns)

        inputs = jnp.array([0.5, 0.5])
        target = jnp.array([1.0])

        def single_loss(n, c):
            transformed = g.transform(state, n, c)
            out = g.forward(state, transformed, inputs)
            return jnp.mean((out - target) ** 2)

        def batch_loss(bn, bc):
            return jnp.sum(jax.vmap(single_loss)(bn, bc))

        grad_fn = jax.jit(jax.grad(batch_loss, argnums=(0, 1)))
        gn, gc = grad_fn(batch_nodes, batch_conns)

        for i in range(4):
            valid_n = jnp.isfinite(gn[i]) | jnp.isnan(batch_nodes[i])
            valid_c = jnp.isfinite(gc[i]) | jnp.isnan(batch_conns[i])
            assert jnp.all(valid_n), f"Network {i}: node grads contain NaN"
            assert jnp.all(valid_c), f"Network {i}: conn grads contain NaN"


# ---------------------------------------------------------------------------
# 5. NaN propagation: disconnected networks output NaN
# ---------------------------------------------------------------------------

class TestNaNPropagation:

    def test_disconnected_network_outputs_nan(self):
        g = genome.DefaultGenome(num_inputs=2, num_outputs=1, max_nodes=5, max_conns=10)
        state, nodes, conns = _make_genome_and_individual(g, seed=0)

        conns_disconnected = jnp.full_like(conns, jnp.nan)

        inputs = jnp.array([1.0, 0.0])
        out = _forward_fn(g, state, nodes, conns_disconnected, inputs)
        assert jnp.all(jnp.isnan(out)), \
            "Disconnected network should output NaN"

    def test_connected_network_no_nan(self):
        g = genome.DefaultGenome(num_inputs=2, num_outputs=1, max_nodes=5, max_conns=10,
                                 output_transform=ACT.sigmoid)
        state, nodes, conns = _make_genome_and_individual(g, seed=0)
        inputs = jnp.array([1.0, 0.0])

        out = _forward_fn(g, state, nodes, conns, inputs)
        assert jnp.all(jnp.isfinite(out)), \
            "Connected network should not output NaN"


# ---------------------------------------------------------------------------
# 6. BiasNode genome gradient
# ---------------------------------------------------------------------------

class TestBiasNodeGrad:

    def test_grad_finite_bias_node(self):
        from tensorneat.genome import BiasNode
        g = genome.DefaultGenome(num_inputs=3, num_outputs=1, max_nodes=7, max_conns=20,
                                 node_gene=BiasNode(), output_transform=ACT.sigmoid)
        state, nodes, conns = _make_genome_and_individual(g, seed=42)
        inputs = jnp.array([1.0, 0.0, 1.0])

        def loss_fn(n, c):
            transformed = g.transform(state, n, c)
            out = g.forward(state, transformed, inputs)
            return jnp.sum(out ** 2)

        grad_n, grad_c = jax.grad(loss_fn, argnums=(0, 1))(nodes, conns)
        assert jnp.all(jnp.isfinite(grad_n) | jnp.isnan(nodes))
        assert jnp.all(jnp.isfinite(grad_c) | jnp.isnan(conns))


# ---------------------------------------------------------------------------
# 7. RecurrentGenome gradient
# ---------------------------------------------------------------------------

class TestRecurrentGrad:

    def test_grad_finite_recurrent(self):
        g = genome.RecurrentGenome(
            num_inputs=2, num_outputs=1, max_nodes=5, max_conns=10,
            output_transform=ACT.sigmoid, activate_time=3,
        )
        state, nodes, conns = _make_genome_and_individual(g, seed=42)
        inputs = jnp.array([1.0, 0.0])

        if not _is_connected(g, state, nodes, conns, inputs):
            pytest.skip("Initial recurrent network is disconnected with this seed")

        def loss_fn(n, c):
            transformed = g.transform(state, n, c)
            out = g.forward(state, transformed, inputs)
            return jnp.sum(out ** 2)

        grad_n, grad_c = jax.grad(loss_fn, argnums=(0, 1))(nodes, conns)
        assert jnp.all(jnp.isfinite(grad_n) | jnp.isnan(nodes))
        assert jnp.all(jnp.isfinite(grad_c) | jnp.isnan(conns))

    def test_jit_grad_recurrent(self):
        g = genome.RecurrentGenome(
            num_inputs=2, num_outputs=1, max_nodes=5, max_conns=10,
            output_transform=ACT.sigmoid, activate_time=3,
        )
        state, nodes, conns = _make_genome_and_individual(g, seed=42)
        inputs = jnp.array([1.0, 0.0])

        if not _is_connected(g, state, nodes, conns, inputs):
            pytest.skip("Initial recurrent network is disconnected with this seed")

        def loss_fn(n, c):
            transformed = g.transform(state, n, c)
            out = g.forward(state, transformed, inputs)
            return jnp.sum(out ** 2)

        grad_fn = jax.jit(jax.grad(loss_fn, argnums=(0, 1)))
        grad_n, grad_c = grad_fn(nodes, conns)
        assert jnp.all(jnp.isfinite(grad_n) | jnp.isnan(nodes))
        assert jnp.all(jnp.isfinite(grad_c) | jnp.isnan(conns))

    def test_loss_decreases_recurrent(self):
        g = genome.RecurrentGenome(
            num_inputs=2, num_outputs=1, max_nodes=5, max_conns=10,
            output_transform=ACT.sigmoid, activate_time=3,
        )
        state, nodes, conns = _make_genome_and_individual(g, seed=42)
        inputs = jnp.array([1.0, 0.0])
        target = jnp.array([0.9])

        if not _is_connected(g, state, nodes, conns, inputs):
            pytest.skip("Initial recurrent network is disconnected with this seed")

        def loss_fn(n, c):
            transformed = g.transform(state, n, c)
            out = g.forward(state, transformed, inputs)
            return jnp.mean((out - target) ** 2)

        initial_loss = loss_fn(nodes, conns)
        grad_fn = jax.jit(jax.grad(loss_fn, argnums=(0, 1)))

        n, c = nodes, conns
        for _ in range(50):
            gn, gc = grad_fn(n, c)
            gn = jnp.where(jnp.isnan(nodes), 0.0, gn)
            gc = jnp.where(jnp.isnan(conns), 0.0, gc)
            n = n - 0.1 * gn
            c = c - 0.1 * gc

        final_loss = loss_fn(n, c)
        assert final_loss < initial_loss, \
            f"Loss should decrease: {initial_loss} -> {final_loss}"


# ---------------------------------------------------------------------------
# 8. Numerical gradient check (finite differences vs autodiff)
# ---------------------------------------------------------------------------

class TestNumericalGradCheck:

    def test_numerical_vs_autodiff(self):
        g = genome.DefaultGenome(num_inputs=2, num_outputs=1, max_nodes=5, max_conns=10,
                                 output_transform=ACT.sigmoid)
        state, nodes, conns = _make_genome_and_individual(g, seed=42)
        inputs = jnp.array([0.5, -0.3])

        def loss_fn(c):
            transformed = g.transform(state, nodes, c)
            out = g.forward(state, transformed, inputs)
            return jnp.sum(out ** 2)

        auto_grad = jax.grad(loss_fn)(conns)

        eps = 1e-4
        numerical_grad = jnp.zeros_like(conns)
        conns_flat = conns.reshape(-1)
        for idx in range(conns_flat.shape[0]):
            if jnp.isnan(conns_flat[idx]):
                continue
            e = jnp.zeros_like(conns_flat).at[idx].set(eps)
            f_plus = loss_fn((conns_flat + e).reshape(conns.shape))
            f_minus = loss_fn((conns_flat - e).reshape(conns.shape))
            numerical_grad = numerical_grad.at[jnp.unravel_index(idx, conns.shape)].set(
                (f_plus - f_minus) / (2 * eps)
            )

        valid = ~jnp.isnan(conns)
        auto_valid = auto_grad[valid]
        num_valid = numerical_grad[valid]
        max_diff = jnp.max(jnp.abs(auto_valid - num_valid))
        assert max_diff < 1e-2, \
            f"Autodiff and numerical gradients differ too much: max_diff={max_diff}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
