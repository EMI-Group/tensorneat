"""Parity of ``grad``: gradient values against ``jax.grad`` of the same loss.

The loss, spelled out once
--------------------------
``loss(outputs) = sum(outputs ** 2)`` over the ``(batch, num_outputs)`` read-out —
a smooth, bounded function of the attributes that touches every output row, so a
mistake in one edge's message or one node's activation shows up in the values (a
constant loss such as ``sum(outputs)`` would not).  The rewrite traces the same
formula with ``evoxc.numpy`` (``enp.sum(out ** 2)``), the legacy side with
``jax.numpy``; the two callables are different objects computing one expression.

The two gradients are compared like this (verified against the legacy source)
----------------------------------------------------------------------------
The legacy ``BaseGenome.grad`` ``vmap``s ``forward`` over a rank-2 ``inputs``
batch and differentiates **one shared** ``(N, L)``/``(E, L)`` gene matrix, so its
gradient is the sum over samples.  The rewrite differentiates a ``(P, ...)``
payload whose population axis is a plain batch, and ``genome.grad`` scalarises
with ``sum(loss_fn(outputs))`` across that axis.  With the single individual tiled
``S`` times (``./conftest.tile_payload``), copy ``p`` therefore holds only sample
``p``'s contribution, and

    legacy_grad == np.sum(new_grad, axis=0)

is the comparison the tests make.  The scalar *loss* needs no such rule: both are
the full sum over all ``(S, O)`` entries.

Tolerances
----------
``rtol=1e-4, atol=1e-5`` on ``float32``.  Gradients are the ratio of two nearly
cancelling differences (sigmoid' is small where the pre-activation is saturated),
so they carry a few more ulp than the forward values do (``rtol=1e-5,
atol=1e-6`` in ``./test_forward.py``); the loss itself is compared at forward
tolerance.

Scope rules (``../CONTEXT.md``)
------------------------------
Single-option ACT/AGG only: the rewrite evaluates every configured option and
selects, so a non-total option could contribute a gradient through a branch it was
not selected into — a genuine semantic difference the tolerance list excludes by
construction.  Activation id ``-1`` (identity now, clamped by the legacy
``lax.switch``) is never used.  Keys are ``int32`` in the rewrite and therefore not
differentiated at all; the legacy grad has one column per gene column and its key
columns come out as exactly ``0`` (comparisons, ``argsort`` and ``searchsorted``
have no derivative), which the padding test asserts where the layouts line up.
The read-out is kept away from ``0`` on every differentiated payload: the
rewrite's power rule is ``y * b / a`` on the base ``a``, so ``x ** 2`` has a
``0 / 0`` derivative at ``x == 0`` while ``jax`` returns ``0`` there — a loss
singularity, not a genome difference, so no test differentiates at that point.
"""

from __future__ import annotations

import numpy as np
import pytest

pytestmark = [pytest.mark.parity]

pytest.importorskip("jax", reason="parity needs the `bench` extra (jax)")

#: The four XOR rows, one per tiled copy (``S = 4`` samples per individual).
XOR_INPUTS = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]], dtype=np.float32)

#: Forward tolerance for the loss value (it is a forward quantity).
LOSS_RTOL = 1e-5
LOSS_ATOL = 1e-6

#: Gradient tolerance (a gradient carries a few more ulp than a forward value).
GRAD_RTOL = 1e-4
GRAD_ATOL = 1e-5


@pytest.fixture(scope="module")
def evoxc_ops():
    """The ``evoxc`` op namespace the rewrite's traced loss is written in.

    Returns:
        The imported ``evoxc`` package, so a test builds
        ``lambda out: evoxc_ops.numpy.sum(out ** 2)`` — a traced formula, never a
        call on numpy data (``support.runner`` remains the only ``build``/``run``
        entry point).
    """
    from support.lazy import require

    return require("evoxc")


def _jax_numpy():
    """Resolve ``jax.numpy`` for the legacy loss, skipping a numpy-only environment."""
    return pytest.importorskip("jax.numpy", reason="parity needs the `bench` extra (jax)")


def _new_loss(evoxc_ops):
    """Build ``sum(out ** 2)`` out of evoxc ops (``out`` is a traced ``(P, O)``)."""

    def loss_fn(outputs):
        return evoxc_ops.numpy.sum(outputs**2)

    return loss_fn


def _legacy_loss(jnp):
    """The same expression in ``jax.numpy`` for the legacy genome."""

    def loss_fn(outputs):
        return jnp.sum(outputs**2)

    return loss_fn


def test_loss_value_matches_jax(
    payloads, genome_pair, new_schedule, new_grad, legacy_grad, tile_payload, evoxc_ops
):
    """Invariant: the differentiated loss equals the legacy loss on the same genome and samples.

    Oracle: ``jax.grad``'s value half.  ``payloads.chain_payload()`` tiled four
    times (one copy per XOR sample) with ``inputs`` the four XOR rows, loss
    ``sum(outputs ** 2)`` on both sides.  Nothing about the gradient enters this
    assertion, so a matching loss with differing gradients points at the
    differentiation, not at the forward pass.

    Assertion: ``np.isclose(new_loss, legacy_loss, rtol=1e-5, atol=1e-6)`` — the
    forward tolerance, since the loss is a forward quantity — and the loss is
    finite on both sides.
    """
    jnp = _jax_numpy()
    pair = genome_pair()
    payload = payloads.chain_payload()
    tiled = tile_payload(payload, len(XOR_INPUTS))
    transformed = new_schedule(pair, tiled)

    loss, _ = new_grad(pair, tiled, transformed, XOR_INPUTS, _new_loss(evoxc_ops))
    legacy_loss, _ = legacy_grad(pair, payload, XOR_INPUTS, _legacy_loss(jnp))
    loss = np.asarray(loss)
    legacy_loss = np.asarray(legacy_loss)

    assert np.all(np.isfinite(loss))
    assert np.all(np.isfinite(legacy_loss))
    assert np.isclose(loss, legacy_loss, rtol=LOSS_RTOL, atol=LOSS_ATOL)


def test_conn_attr_gradients_match_jax(
    payloads,
    genome_pair,
    convert,
    new_schedule,
    new_grad,
    legacy_grad,
    tile_payload,
    evoxc_ops,
):
    """Invariant: the connection-weight gradient equals ``jax.grad``'s, summed over the tiled population.

    Oracle: ``jax.grad`` of the same loss through the legacy genome on the same
    payload.  ``payloads.chain_payload(weight=0.7)`` — a weight that is neither
    ``0`` nor ``1``, so the gradient is not accidentally symmetric — tiled four
    times, XOR inputs.

    Assertion: ``np.allclose(np.sum(new_conn_grad, axis=0),
    np.asarray(legacy_conn_grad), rtol=1e-4, atol=1e-5)`` on the weight columns —
    the axis-0 sum is the "shared parameters" rule from the module docstring — and
    every absent connection row holds exactly ``0`` on both sides (the legacy
    ``where(isnan(conns), 0, grad)`` and the rewrite's padding mask agree), so a
    gradient that leaked into padding fails here too.
    """
    jnp = _jax_numpy()
    pair = genome_pair()
    payload = payloads.chain_payload(weight=0.7)
    tiled = tile_payload(payload, len(XOR_INPUTS))
    transformed = new_schedule(pair, tiled)

    _, (_, conn_grad) = new_grad(
        pair, tiled, transformed, XOR_INPUTS, _new_loss(evoxc_ops)
    )
    _, (_, grad_conns) = legacy_grad(pair, payload, XOR_INPUTS, _legacy_loss(jnp))
    summed = np.sum(np.asarray(conn_grad), axis=0)
    grad_conns = np.asarray(grad_conns)

    _, conns = convert.to_legacy(payload)
    absent = np.isnan(conns[:, 0])
    assert np.array_equal(absent, tiled.conn_keys[0, :, 0] == payloads.I_INF)
    assert np.any(absent) and np.any(~absent), "the chain must have live and padded rows"

    assert np.allclose(
        summed[~absent, 0], grad_conns[~absent, 2], rtol=GRAD_RTOL, atol=GRAD_ATOL
    )
    assert np.all(summed[absent, 0] == 0)
    assert np.all(grad_conns[absent] == 0)


def test_node_attr_gradients_match_jax(
    payloads,
    genome_pair,
    convert,
    new_schedule,
    new_grad,
    legacy_grad,
    tile_payload,
    evoxc_ops,
):
    """Invariant: the node-attribute gradient (bias, response) equals ``jax.grad``'s.

    Oracle: ``jax.grad`` through the legacy genome.  Same chain, tiled four times,
    XOR inputs, with a non-zero ``bias`` and a ``response`` other than ``1`` on the
    hidden rows so ``bias`` and ``response`` cannot be confused in the attribute
    row (the rewrite reads the columns by ``attr_specs`` position, the legacy gene
    by its fixed tuple layout).

    Assertion: ``np.allclose(np.sum(new_node_grad, axis=0)[:, :2],
    legacy_node_grad[:, 1:3], rtol=1e-4, atol=1e-5)`` — note the column offset:
    the legacy node matrix carries the key in column ``0`` (its gradient is ``0``),
    while the rewrite's ``node_attrs`` holds ``bias``/``response`` from column
    ``0`` — and the *integer* columns (aggregation, activation) are exactly ``0``
    on both sides, because they are indices, not parameters.
    """
    jnp = _jax_numpy()
    pair = genome_pair()
    payload = payloads.chain_payload()
    for row in range(pair.num_inputs, pair.max_nodes - pair.num_outputs):
        key = int(payload.node_keys[0, row])
        if key == payloads.I_INF:
            continue
        payloads.set_node(payload, 0, row, key, attrs=(0.3, 1.5, 0.0, 0.0))
    tiled = tile_payload(payload, len(XOR_INPUTS))
    transformed = new_schedule(pair, tiled)

    _, (node_grad, _) = new_grad(
        pair, tiled, transformed, XOR_INPUTS, _new_loss(evoxc_ops)
    )
    _, (grad_nodes, _) = legacy_grad(pair, payload, XOR_INPUTS, _legacy_loss(jnp))
    summed = np.sum(np.asarray(node_grad), axis=0)
    grad_nodes = np.asarray(grad_nodes)

    assert np.allclose(
        summed[:, :2], grad_nodes[:, 1:3], rtol=GRAD_RTOL, atol=GRAD_ATOL
    )
    assert np.all(summed[:, 2:] == 0)
    assert np.all(grad_nodes[:, 0] == 0)
    assert np.all(grad_nodes[:, 3:] == 0)


def test_padded_rows_carry_no_gradient(
    payloads,
    genome_pair,
    convert,
    new_schedule,
    new_grad,
    legacy_grad,
    tile_payload,
    evoxc_ops,
):
    """Invariant: absent rows (``I_INF`` keys, ``NaN`` attributes) have exactly zero gradient.

    Oracle: the two documented clean-ups — the legacy ``jnp.where(jnp.isnan(nodes),
    0.0, grad_nodes)`` and the rewrite's ``select(isnan(g), 0, g)`` plus its
    padding mask.  ``payloads.isolated_payload()`` leaves six node rows and
    fourteen connection rows absent, so most of both matrices is padding, and a
    ``NaN`` anywhere in it would poison the next mutation.

    The output row gets a non-zero ``bias`` so the read-out (which is the
    pre-activation on an output row, and here reads a lone unconnected node) never
    equals ``0``: the loss differentiation must be compared at a point where the
    loss is differentiable, not at the ``x ** 2`` singularity.

    Assertion: every entry of ``new_node_grad``/``new_conn_grad`` on an absent row
    is ``0`` (bit-exact), no entry is ``NaN``, and the same rows of the legacy
    gradient are ``0`` too — so the two clean-ups are shown to agree instead of
    being assumed equivalent.
    """
    jnp = _jax_numpy()
    pair = genome_pair()
    payload = payloads.isolated_payload()
    output_row = pair.max_nodes - pair.num_outputs
    payloads.set_node(
        payload,
        0,
        output_row,
        int(payload.node_keys[0, output_row]),
        attrs=(0.5, 1.0, 0.0, 0.0),
    )
    tiled = tile_payload(payload, len(XOR_INPUTS))
    transformed = new_schedule(pair, tiled)

    _, (node_grad, conn_grad) = new_grad(
        pair, tiled, transformed, XOR_INPUTS, _new_loss(evoxc_ops)
    )
    _, (grad_nodes, grad_conns) = legacy_grad(
        pair, payload, XOR_INPUTS, _legacy_loss(jnp)
    )
    node_grad = np.asarray(node_grad)
    conn_grad = np.asarray(conn_grad)
    grad_nodes = np.asarray(grad_nodes)
    grad_conns = np.asarray(grad_conns)

    absent_nodes = tiled.node_keys[0] == payloads.I_INF
    absent_conns = tiled.conn_keys[0, :, 0] == payloads.I_INF
    assert np.any(absent_nodes) and np.any(absent_conns)

    assert np.all(node_grad[:, absent_nodes, :] == 0)
    assert np.all(conn_grad[:, absent_conns, :] == 0)
    assert not np.any(np.isnan(node_grad))
    assert not np.any(np.isnan(conn_grad))

    nodes, conns = convert.to_legacy(payload)
    legacy_absent_nodes = np.isnan(nodes[:, 0])
    legacy_absent_conns = np.isnan(conns[:, 0])
    assert np.array_equal(legacy_absent_nodes, absent_nodes)
    assert np.array_equal(legacy_absent_conns, absent_conns)
    assert np.all(grad_nodes[legacy_absent_nodes] == 0)
    assert np.all(grad_conns[legacy_absent_conns] == 0)


def test_skipped_node_gradients_match_jax(
    payloads,
    genome_pair,
    convert,
    new_schedule,
    new_grad,
    legacy_grad,
    tile_payload,
    evoxc_ops,
):
    """Invariant: on a graph with a bypassed node the gradients still match ``jax.grad``.

    Oracle: ``jax.grad`` on ``payloads.skip_payload()`` (``0->2``, ``1->2``,
    ``0->3``, ``3->7``) tiled four times, XOR inputs.  Row ``2`` reaches no output,
    so its incoming weights must have a zero gradient on both sides, while row
    ``3``'s do not — the structural case that a "differentiate everything
    uniformly" bug cannot pass, and the one place the depth-unrolled schedule's
    correctness is checked through a derivative rather than through a value.

    Assertion: ``np.allclose(np.sum(new_node_grad, axis=0)[:, :2],
    legacy_node_grad[:, 1:3], rtol=1e-4, atol=1e-5)`` for the node attributes and
    the same axis-0-summed comparison for the weights, plus an explicit
    ``new_conn_grad[:, :2] == 0`` on the two edges into the bypassed row.
    """
    jnp = _jax_numpy()
    pair = genome_pair()
    payload = payloads.skip_payload()
    tiled = tile_payload(payload, len(XOR_INPUTS))
    transformed = new_schedule(pair, tiled)

    _, (node_grad, conn_grad) = new_grad(
        pair, tiled, transformed, XOR_INPUTS, _new_loss(evoxc_ops)
    )
    _, (grad_nodes, grad_conns) = legacy_grad(
        pair, payload, XOR_INPUTS, _legacy_loss(jnp)
    )
    summed_nodes = np.sum(np.asarray(node_grad), axis=0)
    summed_conns = np.sum(np.asarray(conn_grad), axis=0)
    grad_nodes = np.asarray(grad_nodes)
    grad_conns = np.asarray(grad_conns)

    assert np.allclose(
        summed_nodes[:, :2], grad_nodes[:, 1:3], rtol=GRAD_RTOL, atol=GRAD_ATOL
    )

    _, conns = convert.to_legacy(payload)
    absent = np.isnan(conns[:, 0])
    assert np.allclose(
        summed_conns[~absent, 0],
        grad_conns[~absent, 2],
        rtol=GRAD_RTOL,
        atol=GRAD_ATOL,
    )

    bypassed = pair.num_inputs  # the first hidden row, unreachable from the output
    into_bypassed = tiled.conn_keys[0, :, 1] == bypassed
    assert np.count_nonzero(into_bypassed) == pair.num_inputs
    assert np.all(summed_conns[into_bypassed, 0] == 0)
    assert np.all(grad_conns[into_bypassed, 2] == 0)
