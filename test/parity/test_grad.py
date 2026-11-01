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
"""

from __future__ import annotations

import pytest

pytestmark = [pytest.mark.parity]


def test_loss_value_matches_jax(payloads, genome_pair, new_schedule, new_grad, legacy_grad):
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
    pytest.skip(
        "not implemented: the differentiated loss equals the legacy loss "
        "(owner: genome/base.BaseGenome.grad, genome/default.DefaultGenome.forward)"
    )


def test_conn_attr_gradients_match_jax(payloads, genome_pair, new_schedule, new_grad, legacy_grad):
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
    pytest.skip(
        "not implemented: conn attribute gradients equal jax.grad of the same loss "
        "(owner: genome/base.BaseGenome.grad)"
    )


def test_node_attr_gradients_match_jax(payloads, genome_pair, new_schedule, new_grad, legacy_grad):
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
    pytest.skip(
        "not implemented: node attribute gradients equal jax.grad of the same loss "
        "(owner: genome/base.BaseGenome.grad, genome/gene/node/default.DefaultNode.combine)"
    )


def test_padded_rows_carry_no_gradient(payloads, genome_pair, new_schedule, new_grad, legacy_grad):
    """Invariant: absent rows (``I_INF`` keys, ``NaN`` attributes) have exactly zero gradient.

    Oracle: the two documented clean-ups — the legacy ``jnp.where(jnp.isnan(nodes),
    0.0, grad_nodes)`` and the rewrite's ``select(isnan(g), 0, g)`` plus its
    padding mask.  ``payloads.isolated_payload()`` leaves six node rows and
    fourteen connection rows absent, so most of both matrices is padding, and a
    ``NaN`` anywhere in it would poison the next mutation.

    Assertion: every entry of ``new_node_grad``/``new_conn_grad`` on an absent row
    is ``0`` (bit-exact), no entry is ``NaN``, and the same rows of the legacy
    gradient are ``0`` too — so the two clean-ups are shown to agree instead of
    being assumed equivalent.
    """
    pytest.skip(
        "not implemented: padding rows carry exactly zero gradient, never NaN "
        "(owner: genome/base.BaseGenome.grad)"
    )


def test_skipped_node_gradients_match_jax(payloads, genome_pair, new_schedule, new_grad, legacy_grad):
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
    pytest.skip(
        "not implemented: gradients through a bypassed node match jax.grad "
        "(owner: genome/base.BaseGenome.grad, genome/default.DefaultGenome.forward)"
    )
