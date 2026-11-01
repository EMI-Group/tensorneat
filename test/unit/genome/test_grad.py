"""Unit tests for ``BaseGenome.grad`` — the differentiated forward pass.

What this module proves
-----------------------
``grad`` is a **graph factory**, not a kernel: it traces one forward pass, takes
reverse-mode derivatives with respect to the float attribute tensors, and returns
the traced ``evoxc.Graph`` for the pipeline to cache per shape signature. The
tests therefore check three separate things:

* the mathematics — the returned gradients agree with central finite differences
  of the same loss through ``support.oracle.forward``, on a genome small enough to
  perturb one attribute column at a time;
* the evoxc rules the factory has to respect — the loss is scalarized as
  ``sum(loss_fn(outputs))`` (which must not change a per-individual gradient), only
  ``node_attrs``/``conn_attrs`` are differentiated (an ``int32`` differentiated
  argument is a hard ``TransformError``), and no ``evoxc.random`` op may appear
  inside the differentiated graph;
* the NaN contract — padding never carries ``NaN`` into a gradient, because a
  zeroed cotangent times a ``NaN`` local derivative is still ``NaN``, which would
  poison the whole attribute row of that individual.

Oracles used here (never the kernel itself)
-------------------------------------------
* **central finite differences** on ``support.oracle.forward``: the loss is a
  plain numpy function of the attributes, so ``(L(+h) - L(-h)) / 2h`` on one
  coordinate is an independent derivative. ``h`` is chosen relative to the
  attribute magnitude and the comparison uses a relative tolerance, because a
  forward pass of several levels amplifies both terms;
* **closed form** for the bias-only network (all connection weights zero), where
  the read-out is literally the bias and both partial derivatives can be written
  down;
* ``sum`` linearity for the scalarization check, and shape/dtype equality for the
  structural check.

Gotchas this module encodes
---------------------------
* ``grad`` takes an **already-derived** ``Transformed``: ``argsort`` has no AD
  rule, so the schedule must stay outside the differentiated graph. That is why
  every test derives the schedule first (``run_transform``) and hands it in.
* The scalarization is ``sum(loss_fn(outputs))`` over the whole ``(P, ...)`` loss,
  not a mean: the population axis is independent of the sum, so per-individual
  gradients are unchanged — but a test that compared *absolute* loss values
  instead of gradients would see the factor ``P``.
* A node whose value is ``NaN`` (no live incoming edge) contributes a cotangent of
  exactly ``0``, not ``NaN``: the sanitization happens in ``_node_step`` and the
  post-pass ``select(isnan(g), 0, g)`` catches the rest. Both layers are asserted,
  because either one alone leaves a hole.
* Gradients are taken with respect to the *float* attribute tensors only; the
  keys and the schedule are non-differentiated ``int32``/``bool`` inputs of the
  same graph. A gradient tuple of length two (node attributes, connection
  attributes) is therefore the contract, not an implementation detail.

Backend is ``"numpy"`` throughout, through the shared runner only; no
``jax``/``torch``, no network, no GPU.
"""

from __future__ import annotations

import pytest


# --------------------------------------------------------------------------- #
# the factory contract
# --------------------------------------------------------------------------- #


def test_grad_returns_a_graph_factory_not_a_value(
    genome, payloads, runner, run_transform
):
    """Invariant: ``grad`` returns an ``evoxc.Graph`` and does not evaluate anything.

    Oracle: the type of the returned object (an ``evoxc.Graph``, i.e. the same
    kind of value ``evoxc.grad`` produces) and the fact that building it consumes
    no loss call. ``BaseGenome.grad`` documents the graph as mapping the same
    tensor arguments to ``(scalar_loss, (node_attr_grads, conn_attr_grads))``, so
    ``support.runner.run_graph(graph, node_keys, node_attrs, conn_keys,
    conn_attrs, inputs)`` is the only way to get numbers out — a kernel returning
    an array would fail here instead of silently working.
    """
    pytest.skip(
        "not implemented: grad returns a traced evoxc.Graph, not a value "
        "(owner: genome/base.BaseGenome.grad)"
    )


def test_grad_graph_contains_no_random_op_and_is_deterministic(
    payloads, runner, run_transform
):
    """Invariant: the differentiated graph is random-free, hence run-to-run identical.

    Oracle: two runs of the *same* compiled graph on the *same* inputs return
    bit-identical gradients and loss. The observable form of "no random op inside
    the differentiated graph" is exactly this: an ``evoxc.random`` draw inside a
    differentiated region is a hard ``evoxc.TransformError`` at lowering time, so a
    graph that lowers and then reproduces its own output cannot contain one. Any
    sampling the genome needs (initialization, mutation) belongs to another graph.
    """
    pytest.skip(
        "not implemented: the differentiated graph lowers and is deterministic "
        "(owner: genome/base.BaseGenome.grad)"
    )


def test_gradient_shapes_and_dtypes_follow_the_differentiated_inputs(
    payloads, runner, run_transform
):
    """Invariant: the gradient pair has the shapes and dtypes of ``node_attrs``/``conn_attrs``.

    Oracle: structural equality with the differentiated inputs —
    ``(P, N, NA) float32`` and ``(P, E, CA) float32`` — and a tuple of exactly two
    arrays, because keys and the schedule stay non-differentiated ``int32``/``bool``
    graph inputs. A float64 or broadcast gradient silently changes what the
    algorithm's optimizer does, so dtype is part of the contract here.
    """
    pytest.skip(
        "not implemented: gradient shapes/dtypes mirror node_attrs and conn_attrs "
        "(owner: genome/base.BaseGenome.grad)"
    )


# --------------------------------------------------------------------------- #
# finite differences
# --------------------------------------------------------------------------- #


def test_conn_attr_gradient_matches_finite_differences(
    payloads, oracle, runner, run_transform, run_forward
):
    """Invariant: ``d loss / d conn_attrs`` equals central differences of the same loss.

    Oracle: ``support.oracle.forward`` as a plain numpy function of the weight
    column — perturb one attribute entry by ``±h``, recompute the loss through the
    oracle, and compare ``(L(+h) - L(-h)) / 2h`` with the kernel's gradient entry
    under a relative tolerance. A single-output, single-activation, shallow genome
    keeps the difference well-conditioned; the data for the finite difference is
    *never* taken from the kernel, so the two derivatives are independent
    computations.
    """
    pytest.skip(
        "not implemented: connection-weight gradient equals central finite differences "
        "(owner: genome/base.BaseGenome.grad, genome/base.BaseGenome._node_step)"
    )


def test_node_attr_gradient_matches_finite_differences(
    payloads, oracle, runner, run_transform, run_forward
):
    """Invariant: ``d loss / d node_attrs`` equals central differences of the same loss.

    Oracle: the same finite-difference procedure applied to the ``bias``/``response``
    columns. The ``aggregation``/``activation`` columns are option *ids* stored as
    floats: they have no derivative, so they are held constant and their gradient
    entries are asserted to be finite (``select(isnan(g), 0, g)``), never to match
    a difference quotient.
    """
    pytest.skip(
        "not implemented: node-attribute gradient equals central finite differences "
        "(owner: genome/base.BaseGenome.grad)"
    )


def test_bias_only_network_gradient_is_closed_form(
    genome, payloads, runner, run_transform
):
    """Invariant: with zero weights the read-out is the bias and both gradients are closed form.

    Oracle: closed form. ``DefaultConn``'s message is ``weight * source``, so with
    every weight ``0`` a node computes ``bias + response * 0 = bias`` regardless of
    its sources; with the identity activation and ``response = 1`` the read-out is
    exactly the output row's ``bias``. For ``L = loss(readout)`` that gives
    ``dL/dbias = dL/dreadout`` and ``dL/dweight = dL/dreadout * source_value`` —
    two expressions a test can state without any reference implementation. This is
    the case where the *product* rule through the weight is still exercised while
    the value path is trivial.
    """
    pytest.skip(
        "not implemented: zero-weight network gradient in closed form "
        "(owner: genome/base.BaseGenome.grad, genome/gene/conn/default.DefaultConn.forward)"
    )


# --------------------------------------------------------------------------- #
# NaN and padding
# --------------------------------------------------------------------------- #


def test_nan_padding_never_poisons_a_gradient(
    payloads, runner, run_transform, run_forward
):
    """Invariant: a padded or dead node contributes a cotangent of ``0``, not ``NaN``.

    Oracle: the documented cleaning rule. ``node_attrs``/``conn_attrs`` padding
    rows are ``NaN``, and a present node with no live incoming edge holds a ``NaN``
    *value*; both go through ``select(isnan(g), 0, g)`` plus a padding mask, so
    every returned gradient entry is finite. The test asserts finiteness over the
    whole gradient pair — on a payload that deliberately mixes padding rows, an
    isolated node and a live network — rather than on selected entries.
    """
    pytest.skip(
        "not implemented: NaN padding and dead nodes yield zero, finite gradients "
        "(owner: genome/base.BaseGenome.grad, genome/base.BaseGenome._node_step)"
    )


def test_padded_rows_receive_a_zero_gradient(payloads, runner, run_transform):
    """Invariant: gradient entries on absent rows are exactly ``0``.

    Oracle: the padding mask. Rows whose ``node_keys``/``conn_keys`` are ``I_INF``
    (and whose attribute rows are ``NaN``) never take part in the forward pass, so
    their gradient is masked to ``0`` — not to ``NaN`` and not to a finite value
    that an optimizer would then apply to a nonexistent gene.
    """
    pytest.skip(
        "not implemented: gradient is masked to zero on absent rows "
        "(owner: genome/base.BaseGenome.grad)"
    )


# --------------------------------------------------------------------------- #
# scalarization and the population axis
# --------------------------------------------------------------------------- #


def test_scalarized_loss_leaves_per_individual_gradients_unchanged(
    payloads, oracle, runner, run_transform
):
    """Invariant: the population sum in ``sum(loss_fn(outputs))`` does not couple individuals.

    Oracle: finite differences per individual (as in
    :func:`test_conn_attr_gradient_matches_finite_differences`) compared against the
    gradients of one ``P == 4`` run. The scalarization adds a constant multiple of
    the other members' losses to the objective, whose derivative with respect to
    *this* member's attributes is zero, so every entry must be unchanged. It also
    pins the sign of the accumulated term: a mean instead of a sum would scale
    every gradient by ``1/P`` and fail the relative comparison.
    """
    pytest.skip(
        "not implemented: sum(loss) scalarization keeps per-individual gradients "
        "(owner: genome/base.BaseGenome.grad)"
    )
