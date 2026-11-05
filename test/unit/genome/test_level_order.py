"""Unit tests for level-order scheduling: a row index carries no evaluation order.

What this module proves
-----------------------
Every payload ``test/support/payloads.py`` can build — and both random-DAG
generators in the suite — writes each connection with
``source_row < target_row``.  The level-synchronous forward sweep and the dense
gradient pass are therefore only ever exercised where topological order and row
order agree, so an implementation that read a node's level off its row index (or
merely swept rows in index order) would pass the rest of the suite untouched.

This module removes that coincidence.  It writes a hand-built chain whose rows
are in topological order, then evaluates an **isomorphic relabel** of the same
genome in which the chain runs *against* the row order: all four joints between
consecutive hidden nodes point from a higher row to a lower one.  Three
independent facts are pinned about the relabel:

* ``Transformed.node_level`` is the longest-path depth derived from the
  connection *keys*; it moves with its node and never with the row index.
* ``Genome.forward`` returns the same value for every node, bit for bit.  Every
  hidden node is read out through a probe output row permanently wired to that
  node's key, so the read-out index is the node's canonical index in both
  payloads and no re-ordering is needed on this side.
* The dense gradient pass (``Genome.grad``) returns the same per-gene gradient,
  bit for bit: the connection gradient entry for entry (a connection row holds
  the same gene in both payloads, only its endpoint keys move), the node
  gradient after inverting the node-row permutation.

A second, hand-computable variant isolates a single edge.  ``0 -> 2 -> 1 -> out``
puts the consumer at row ``1`` and its source at row ``2``; the closed form
``36.25`` is reachable only if row ``2`` is evaluated before row ``1``.

Why a row-order-dependent kernel cannot pass
--------------------------------------------
``node_level`` is the mask the forward sweep uses (``update_mask =
node_level == level``) and the static level masks of the dense pass.  A schedule
that read the level off the row index would update the level-5 node now sitting
at row ``2`` at step 2 — before its source at row ``3`` has a value — and never
revisit it, so the probe read-outs would be ``NaN`` instead of the closed form
below (``NaN`` never compares equal to a value, bytewise or otherwise).

Oracles used here (never the implementation's own intermediate state)
--------------------------------------------------------------------
* **hand arithmetic**, written out below: every value is a dyadic rational, so
  it is exact in ``float32`` and asserted with ``array_equal`` / ``tobytes``;
* **the canonical payload's own result**, an independent run of the same algebra
  in which row order *does* follow level order — a row-order-dependent kernel
  makes the two sides disagree;
* **the documented schedule invariants**: the level of an edge's destination is
  strictly greater than the level of its source, read off the permuted schedule.

Gotchas this module encodes
---------------------------
* ``forward`` has no per-node output.  Intermediate values are read out through
  *probe* output rows (an edge of weight ``1`` from the hidden row into an output
  row), exactly as ``test_forward.py`` does; output rows are read out
  unactivated, so a probe row with the neutral ``(bias=0, response=1)``
  attribute row returns its source value unchanged.
* A node's key equals its row in every payload builder here, and the relabel
  moves the key *with* the node (row ``old`` -> row ``new``, key ``old`` -> key
  ``new``).  Input keys must stay on rows ``0..num_inputs-1`` and output keys on
  the last ``num_outputs`` rows (the row/key invariant the read-out depends on),
  so only hidden rows are permuted.
* ``grad`` takes an already-derived ``Transformed`` (``argsort`` has no AD rule)
  and returns a fresh ``evoxc.Graph`` per call, which ``support.runner.run_graph``
  compiles and runs — the same path as ``test_grad.py``.

Backend is ``"numpy"`` throughout, through the shared runner only; no
``jax``/``torch``, no network, no GPU.
"""

from __future__ import annotations

import numpy as np
import pytest

# --------------------------------------------------------------------------- #
# geometry of the probe chain
# --------------------------------------------------------------------------- #

#: Two inputs, five hidden nodes and six output rows: five probe read-outs plus
#: the chain's real read-out, so every hidden node is observable.
NUM_INPUTS = 2
NUM_OUTPUTS = 6
MAX_NODES = 13
MAX_CONNS = 16

#: The activation id that means "identity" (never an option index), so a hidden
#: value is the plain affine composition ``bias + response * agg``.
IDENTITY = -1.0

#: Hidden rows of the canonical payload, in topological order.
HIDDEN_ROWS = (2, 3, 4, 5, 6)

#: Canonical hidden row -> hidden row in the permuted payload: the reversal, so
#: the chain runs from row ``6`` down to row ``2``.  An involution, and the only
#: permutation used here.
REVERSAL = {2: 6, 3: 5, 4: 4, 5: 3, 6: 2}

#: ``canonical hidden row -> (bias, response)`` of the chain.
HIDDEN_ATTRS = {
    2: (0.5, 1.5),
    3: (-0.25, 2.0),
    4: (0.125, 0.5),
    5: (0.75, 1.0),
    6: (-0.5, 1.5),
}

#: ``(bias, response)`` of the chain's real output row; the response scales the
#: read-out edge, which is what makes its closed-form gradient distinguishable.
FINAL_ATTRS = (0.25, 2.0)

#: Every chain and probe edge carries this weight, so a value equals the value
#: of its source (times the node's own response).
CHAIN_WEIGHT = 1.0

#: One input row: ``x0 + x1 = 0.25`` (dyadic, exact in float32).
INPUTS = np.array([[0.5, -0.25]], dtype=np.float32)

#: Hand arithmetic on the canonical chain (identity activations, weight ``1``)::
#:
#:     v2 =  0.5   + 1.5 * (x0 + x1)          = 0.875
#:     v3 = -0.25  + 2.0 * v2                 = 1.5
#:     v4 =  0.125 + 0.5 * v3                 = 0.875
#:     v5 =  0.75  + 1.0 * v4                 = 1.625
#:     v6 = -0.5   + 1.5 * v5                 = 1.9375
#:     out_0..out_4 = v2..v6                  (probe rows, neutral attributes)
#:     out_5 = 0.25 + 2.0 * v6                = 4.125
CANONICAL_OUTPUT = np.array(
    [[0.875, 1.5, 0.875, 1.625, 1.9375, 4.125]], dtype=np.float32
)

#: Hand arithmetic on the same chain's ``node_level``: inputs at ``0``, the five
#: hidden nodes at ``1..5``, the probe output rows one level below their source
#: and the real output row at ``6``.
CANONICAL_LEVELS = np.array([0, 0, 1, 2, 3, 4, 5, 2, 3, 4, 5, 6, 6], dtype=np.int32)

#: Target of the quadratic loss every gradient assertion differentiates.
TARGET = 0.5


# --------------------------------------------------------------------------- #
# payload builders
# --------------------------------------------------------------------------- #


def _chain(payloads, rows):
    """Build the probe chain with canonical hidden row ``h`` written at ``rows[h]``.

    The graph is always the same one: both inputs feed the first hidden node, the
    hidden nodes form a chain, every hidden node has a weight-``1`` edge into its
    own probe output row, and the last hidden node feeds the real read-out.  Only
    the *rows* the hidden nodes occupy change.

    Args:
        payloads: The ``support.payloads`` module.
        rows: ``canonical hidden row -> payload row``, a bijection of
            :data:`HIDDEN_ROWS`.  Every present node's key equals its row.

    Returns:
        The one-individual ``Payload``.  Connection rows keep one fixed order
        (six chain edges, five probe edges, one read-out edge), so connection row
        ``e`` holds the same gene in every payload built here; only its endpoint
        keys move with their node.
    """
    payload = payloads.new_payload(1, MAX_NODES, MAX_CONNS)
    payloads.set_node(payload, 0, 0, 0)
    payloads.set_node(payload, 0, 1, 1)
    for old in HIDDEN_ROWS:
        bias, response = HIDDEN_ATTRS[old]
        row = rows[old]
        payloads.set_node(payload, 0, row, row, (bias, response, 0.0, IDENTITY))
    for row in range(7, 12):  # one probe read-out per hidden node
        payloads.set_node(payload, 0, row, row)
    payloads.set_node(payload, 0, 12, 12, (*FINAL_ATTRS, 0.0, IDENTITY))

    edges = [(0, rows[2], CHAIN_WEIGHT), (1, rows[2], CHAIN_WEIGHT)]
    edges += [
        (rows[source], rows[target], CHAIN_WEIGHT)
        for source, target in zip(HIDDEN_ROWS, HIDDEN_ROWS[1:])
    ]
    edges += [
        (rows[old], 7 + index, CHAIN_WEIGHT)
        for index, old in enumerate(HIDDEN_ROWS)
    ]
    edges.append((rows[6], 12, CHAIN_WEIGHT))
    for conn_row, (source, target, weight) in enumerate(edges):
        payloads.set_conn(payload, 0, conn_row, source, target, (weight,))
    return payload


def _present_edges(payloads, payload, relabel=None):
    """The ``(source row, target row, weight)`` of every present connection row.

    Args:
        payloads: The ``support.payloads`` module (for the padding sentinel).
        payload: The ``Payload`` to read; only individual ``0`` is inspected.
        relabel: Optional ``row -> row`` map applied to both endpoints, so an
            edge of a permuted payload can be compared with the canonical one.

    Returns:
        A list in connection-row order; padding rows are skipped.
    """
    edges = []
    for row in range(payload.max_conns):
        source = int(payload.conn_keys[0, row, 0])
        if source == payloads.I_INF:
            continue
        target = int(payload.conn_keys[0, row, 1])
        if relabel is not None:
            source, target = relabel[source], relabel[target]
        edges.append((source, target, float(payload.conn_attrs[0, row, 0])))
    return edges


def _inverse(perm):
    """The inverse of a row permutation given as ``old -> new`` (identity elsewhere)."""
    inverse = {}
    for old, new in perm.items():
        inverse[new] = old
    for old in range(MAX_NODES):
        inverse.setdefault(old, old)
    return inverse


def _unpermute(values, perm):
    """``out[old] = values[perm[old]]`` — read a permuted per-row array in canonical order.

    Args:
        values: ``(MAX_NODES,)`` or ``(MAX_NODES, width)`` array of the permuted
            payload; axis ``0`` is the node row.
        perm: ``canonical row -> permuted row`` (identity for rows it omits).

    Returns:
        A new array of the same shape and dtype.
    """
    out = np.empty_like(values)
    for old in range(values.shape[0]):
        out[old] = values[perm.get(old, old)]
    return out


def _loss(outputs):
    """Per-individual quadratic loss ``sum over outputs of (out - TARGET) ** 2``.

    Squared with a ``multiply`` on purpose: evoxc's ``power`` pullback emits
    ``log(base)`` unconditionally, so ``residual ** 2`` would warn on a negative
    residual (``test_grad.py``'s note).
    """
    residual = outputs - TARGET
    return residual * residual


def _gradients(runner, genome_object, state, payload, transformed, inputs):
    """Run ``genome.grad`` of :func:`_loss` and return ``(node_grad, conn_grad)`` numpy."""
    graph = genome_object.grad(state, transformed, *payload.args(), inputs, _loss)
    return runner.run_graph(graph, *payload.args(), inputs)


# --------------------------------------------------------------------------- #
# fixtures: one genome per payload family, shared by the tests of this module
# --------------------------------------------------------------------------- #


@pytest.fixture(scope="module")
def network(payloads):
    """The ``DefaultGenome`` and ``State`` whose statics match every probe chain here.

    A genome owns its schedule statics (``max_nodes``/``max_conns``/``key_span``),
    so the payload of :func:`_chain` must be built for exactly these shapes.
    Module scope lets the ``run_forward`` executable cache serve both payloads
    (same shapes and dtypes) instead of retracing per test.
    """
    genome_object = payloads.default_genome(
        num_inputs=NUM_INPUTS,
        num_outputs=NUM_OUTPUTS,
        max_nodes=MAX_NODES,
        max_conns=MAX_CONNS,
        max_in_degree=4,
    )
    return genome_object, payloads.genome_state(genome_object)


# --------------------------------------------------------------------------- #
# the schedule: levels come from the keys, never from the row index
# --------------------------------------------------------------------------- #


def test_the_schedule_reads_levels_from_the_keys_not_from_the_row_index(
    payloads, network, run_transform
):
    """Invariant: ``node_level`` is the longest-path depth of the *keys*, moved by the relabel.

    Oracle: hand arithmetic on the canonical chain (the literal
    :data:`CANONICAL_LEVELS`), plus the relabel itself.  The permuted payload is
    checked to be the canonical graph up to a renaming first — the same node
    attributes at the relabelled rows and the same edge set with both endpoints
    relabelled — so a builder mistake cannot make the rest of the comparison
    vacuous.  Then two things are pinned:

    * inverting the row permutation on the permuted levels reproduces the
      canonical levels row for row (a level belongs to a node, not to a row);
    * every chain joint that runs against the row order has its source at a
      *lower* level than its destination, and the permuted level vector is not
      the row-index vector — so a schedule that read the level off the row index
      is excluded, not merely untested.
    """
    genome_object, state = network
    identity_rows = {row: row for row in HIDDEN_ROWS}
    canonical = _chain(payloads, identity_rows)
    permuted = _chain(payloads, REVERSAL)

    # Precondition of the experiment: the permuted payload *is* the canonical one
    # under the relabel.  Node attributes sit on the relabelled rows ...
    for old in range(MAX_NODES):
        new = REVERSAL.get(old, old)
        assert (
            permuted.node_attrs[0, new].tobytes()
            == canonical.node_attrs[0, old].tobytes()
        ), f"node attributes of canonical row {old} did not move to row {new}"
    # ... and every edge survives with both endpoints relabelled.
    assert sorted(_present_edges(payloads, permuted, _inverse(REVERSAL))) == sorted(
        _present_edges(payloads, canonical)
    )

    canonical_levels = np.asarray(
        run_transform(genome_object, state, canonical).node_level
    )[0]
    permuted_levels = np.asarray(
        run_transform(genome_object, state, permuted).node_level
    )[0]

    # The canonical chain's longest-path depth, spelled out by hand.
    assert np.array_equal(canonical_levels, CANONICAL_LEVELS)
    # Inverting the permutation on the permuted levels gives it back exactly.
    assert np.array_equal(_unpermute(permuted_levels, REVERSAL), CANONICAL_LEVELS)

    # All four joints between consecutive hidden nodes point from a higher row to
    # a lower one, and each still has its destination one level above its source.
    crossings = [
        (source, target)
        for source, target, _ in _present_edges(payloads, permuted)
        if source > target
    ]
    assert len(crossings) == 4, crossings
    for source, target in crossings:
        assert permuted_levels[source] < permuted_levels[target], (source, target)
    # The level is not the row index anywhere, so "level == row" is excluded.
    assert not np.array_equal(
        permuted_levels, np.arange(MAX_NODES, dtype=np.int32)
    )
    assert np.array_equal(
        permuted_levels, np.array([0, 0, 5, 4, 3, 2, 1, 2, 3, 4, 5, 6, 6], np.int32)
    )


# --------------------------------------------------------------------------- #
# the forward pass: the same value per node, bit for bit
# --------------------------------------------------------------------------- #


def test_forward_values_are_unchanged_by_the_row_relabel(
    payloads, network, run_transform, run_forward
):
    """Invariant: every node keeps its forward value when its row changes.

    Oracle: the hand arithmetic of :data:`CANONICAL_OUTPUT` on the canonical
    payload, then the canonical read-out itself for the permuted payload.  Both
    sides are ``float32`` and every intermediate is a dyadic rational, so the
    comparison is ``tobytes()`` equality — not an allclose that would hide a
    reassociation.

    The permuted chain runs against the row order (``6 -> 5 -> 4 -> 3 -> 2``),
    while each probe output row stays wired to its node's key — so the read-out
    vector is indexed by canonical node in both payloads.  A row-order-dependent
    sweep computes a later-level node before its source and never revisits it,
    turning every probe into ``NaN``; ``NaN`` is what makes this assertion fail.
    """
    genome_object, state = network
    canonical = _chain(payloads, {row: row for row in HIDDEN_ROWS})
    permuted = _chain(payloads, REVERSAL)

    canonical_out = np.asarray(
        run_forward(
            genome_object,
            state,
            run_transform(genome_object, state, canonical),
            canonical,
            INPUTS,
        )
    )
    permuted_out = np.asarray(
        run_forward(
            genome_object,
            state,
            run_transform(genome_object, state, permuted),
            permuted,
            INPUTS,
        )
    )

    np.testing.assert_array_equal(canonical_out, CANONICAL_OUTPUT)
    assert np.all(np.isfinite(permuted_out)), permuted_out
    assert permuted_out.shape == (1, NUM_OUTPUTS)
    assert permuted_out.tobytes() == canonical_out.tobytes()


# --------------------------------------------------------------------------- #
# the dense gradient pass: the same per-gene gradient, bit for bit
# --------------------------------------------------------------------------- #


def test_dense_pass_gradients_are_unchanged_by_the_row_relabel(
    payloads, network, runner, run_transform
):
    """Invariant: ``grad`` returns the same per-gene gradient after the relabel.

    Oracle: the canonical payload's gradient (an independent run of the same
    algebra in which row order follows level order), mapped back through the row
    permutation, plus three closed-form entries — so the permuted side is not
    merely self-consistent with the canonical one:

    * the read-out row's bias gradient is ``dL/dout = 2 * (4.125 - 0.5) = 7.25``;
    * its response gradient is that times the row's aggregate, ``7.25 * 1.9375``;
    * the read-out *connection* weight gradient is ``7.25 * response * 1.9375``
      (the row's response is ``2.0``), and the last probe edge's weight gradient
      is ``2 * (1.9375 - 0.5) * 1.9375``.

    The connection gradient is compared entry for entry: connection row ``e``
    holds the same gene in both payloads (only its endpoint keys moved), so its
    derivative must be identical.  The node gradient lives on rows, so it is
    compared after inverting the permutation.  Padding connection rows must
    gradient to exactly ``0`` in both — the NaN-cleaned attribute boundary.
    """
    genome_object, state = network
    canonical = _chain(payloads, {row: row for row in HIDDEN_ROWS})
    permuted = _chain(payloads, REVERSAL)

    node_canonical, conn_canonical = _gradients(
        runner,
        genome_object,
        state,
        canonical,
        run_transform(genome_object, state, canonical),
        INPUTS,
    )
    node_permuted, conn_permuted = _gradients(
        runner,
        genome_object,
        state,
        permuted,
        run_transform(genome_object, state, permuted),
        INPUTS,
    )

    # Connection gradients: same gene on the same row, so identical bytes.
    assert conn_permuted.shape == conn_canonical.shape == (1, MAX_CONNS, 1)
    assert conn_permuted.tobytes() == conn_canonical.tobytes()
    # Node gradients: a level belongs to a node, so invert the row permutation.
    assert node_permuted.shape == node_canonical.shape == (1, MAX_NODES, 4)
    assert (
        _unpermute(node_permuted[0], REVERSAL).tobytes()
        == node_canonical[0].tobytes()
    )

    # Closed forms on the permuted payload (which routes the chain through
    # rows 6 -> 5 -> 4 -> 3 -> 2).  The real read-out sits on row 12 and its
    # source -- the last chain node, value 1.9375 -- on row 2.
    read_out = np.float32(2.0 * (4.125 - 0.5))
    assert node_permuted[0, 12, 0] == read_out
    assert node_permuted[0, 12, 1] == read_out * np.float32(1.9375)
    assert conn_permuted[0, 11, 0] == read_out * np.float32(2.0 * 1.9375)
    # Probe edge of connection row 10: row 2 (value 1.9375) -> output row 11.
    assert conn_permuted[0, 10, 0] == np.float32(
        2.0 * (1.9375 - 0.5) * 1.9375
    )
    # Padding connection rows (12..15) carry the NaN sentinel and must gradient
    # to exactly zero, never to NaN.
    assert np.all(conn_permuted[0, 12:] == 0.0)
    assert np.all(conn_canonical[0, 12:] == 0.0)


# --------------------------------------------------------------------------- #
# a single crossing edge, with a hand-computable closed form
# --------------------------------------------------------------------------- #

#: Geometry of the isolated variant: one input, two hidden stages and two output
#: rows (one real read-out and one probe of the first hidden stage).
SMALL_MAX_NODES = 5
SMALL_MAX_CONNS = 5

#: ``x0 = 1`` keeps the closed form small.
SMALL_INPUTS = np.array([[1.0]], dtype=np.float32)


def _small_chain(payloads, rows):
    """The three-node chain ``0 -> a -> b -> out`` with its stages on ``rows``.

    Args:
        payloads: The ``support.payloads`` module.
        rows: ``{1: row of stage a, 2: row of stage b}``.  Rows ``3`` (the real
            read-out) and ``4`` (a probe of stage ``a``) are fixed output rows.

    Returns:
        The one-individual payload: weight ``2`` into ``a``, weight ``3`` from
        ``a`` to ``b``, weight ``5`` from ``b`` into the read-out and weight ``1``
        from ``a`` into the probe.
    """
    payload = payloads.new_payload(1, SMALL_MAX_NODES, SMALL_MAX_CONNS)
    payloads.set_node(payload, 0, 0, 0)
    payloads.set_node(payload, 0, rows[1], rows[1], (0.5, 1.0, 0.0, IDENTITY))
    payloads.set_node(payload, 0, rows[2], rows[2], (-0.25, 1.0, 0.0, IDENTITY))
    payloads.set_node(payload, 0, 3, 3)
    payloads.set_node(payload, 0, 4, 4)
    payloads.set_conn(payload, 0, 0, 0, rows[1], (2.0,))
    payloads.set_conn(payload, 0, 1, rows[1], rows[2], (3.0,))
    payloads.set_conn(payload, 0, 2, rows[2], 3, (5.0,))
    payloads.set_conn(payload, 0, 3, rows[1], 4, (1.0,))
    return payload


@pytest.fixture(scope="module")
def small_network(payloads):
    """The ``DefaultGenome`` and ``State`` of the isolated three-node variant."""
    genome_object = payloads.default_genome(
        num_inputs=1,
        num_outputs=2,
        max_nodes=SMALL_MAX_NODES,
        max_conns=SMALL_MAX_CONNS,
    )
    return genome_object, payloads.genome_state(genome_object)


def test_a_lower_row_consumer_reads_the_higher_row_source_previous_level_value(
    payloads, small_network, run_transform, run_forward
):
    """Invariant: a consumer below its source sees the source's value from the previous level.

    Oracle: the closed form of the chain ``0 -> a -> b -> out`` with
    ``x0 = 1``, weight ``2``/``3``/``5`` and the attribute rows written in
    :func:`_small_chain`::

        v_a = 0.5   + 1.0 * (2 * 1)   = 2.5
        v_b = -0.25 + 1.0 * (3 * v_a) = 7.25
        out = 0     + 1.0 * (5 * v_b) = 36.25

    The crossing payload writes ``b`` on row ``1`` and its source ``a`` on row
    ``2``, so the consumer sits *below* its source.  The read-out is ``36.25``
    only if ``a`` is evaluated first, at its own level ``1``, and ``b`` then reads
    that value at level ``2`` — the source's previous-level value.  A sweep that
    followed the row index would meet ``b`` before ``a`` has a value and the
    read-out would be ``NaN``.  The probe output row ``4`` reads ``v_a = 2.5``
    directly, so the intermediate is pinned too, and the ordered layout (rows in
    topological order) must produce the identical two values.
    """
    genome_object, state = small_network
    ordered = _small_chain(payloads, {1: 1, 2: 2})
    crossed = _small_chain(payloads, {1: 2, 2: 1})

    ordered_out = np.asarray(
        run_forward(
            genome_object,
            state,
            run_transform(genome_object, state, ordered),
            ordered,
            SMALL_INPUTS,
        )
    )
    crossed_transformed = run_transform(genome_object, state, crossed)
    crossed_out = np.asarray(
        run_forward(
            genome_object, state, crossed_transformed, crossed, SMALL_INPUTS
        )
    )

    expected = np.array([[36.25, 2.5]], dtype=np.float32)
    np.testing.assert_array_equal(ordered_out, expected)
    np.testing.assert_array_equal(crossed_out, expected)
    assert crossed_out.tobytes() == ordered_out.tobytes()

    # Hand arithmetic on the schedule: ``a`` (row 2) is level 1, its consumer
    # ``b`` (row 1) is level 2, the read-out below ``b`` is level 3 and the probe
    # of ``a`` is level 2.  The lower row carries the *higher* level.
    levels = np.asarray(crossed_transformed.node_level)[0]
    assert np.array_equal(levels, np.array([0, 2, 1, 3, 2], dtype=np.int32))
    assert levels[2] < levels[1]

    # The consumer's incoming window points at the higher row, and only at it.
    in_src = np.asarray(crossed_transformed.in_src)[0]
    in_valid = np.asarray(crossed_transformed.in_valid)[0]
    sources = [int(src) for src, live in zip(in_src[1], in_valid[1]) if live]
    assert sources == [2]
