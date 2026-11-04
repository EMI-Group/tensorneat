"""Unit tests for ``tensorneat.host_payload._one_individual``, the payload funnel of ``Pipeline.show``.

What this module proves
-----------------------
``_one_individual`` is the one place a host-side genome payload is normalised
before ``Pipeline.show`` and the checkpoint path use it, so its contract is the
pair of spellings ``Pipeline.show`` documents, and nothing else:

* a **one-individual batch** -- ``node_keys (1, max_nodes)``,
  ``node_attrs (1, max_nodes, A)``, ``conn_keys (1, max_conns, key_width)``,
  ``conn_attrs (1, max_conns, W)``, i.e. ``state.pop``/``best_genome`` at
  ``P = 1`` -- comes back unchanged, array for array;
* the **bare per-individual rows** a checkpoint stores (``node_keys (max_nodes,)``,
  ``node_attrs (max_nodes, A)``, ``conn_keys (max_conns, key_width)``,
  ``conn_attrs (max_conns, W)``) are promoted with a leading axis of 1 and
  describe the same individual;
* a payload holding **more than one individual** is rejected up front with a
  ``TensorNEATError`` naming both accepted spellings -- not with the
  "arrays disagree about the population axis" message, which describes a
  different bug and would send the caller looking for a broken shape;
* a **missing** payload (``None``, e.g. ``auto_run``'s ``best_genome`` when no
  generation produced a finite fitness) is rejected with an actionable
  ``TensorNEATError`` rather than the bare ``TypeError`` ``list(None)`` raises;
* every pre-existing malformed-input error path keeps its own message (the
  field count, a rank-0 field, ``node_keys`` above rank 2, ``conn_keys`` outside
  rank 2-3, a key/attribute rank mismatch, disagreeing population axes).

Oracles used here (never the implementation's own docstring)
-----------------------------------------------------------
The arrays are built by hand with *distinct values per field and per individual*
(``_payload``), so "the same individual" is a value comparison, not a shape
comparison: a private copy that got reshaped or transposed into the right shape
still fails on the values, and a payload that was silently sliced down to its
first individual is caught by the same comparison. The shapes in the messages
are the documented static signature (``max_nodes``/``max_conns`` constants
below), so an error that stops naming them fails loudly.

No ``Pipeline`` build and no ``jax``/``torch``: the helper is host-only, so these
tests are plain numpy, except :func:`test_run_network_chunks_the_tiled_batch`,
which traces one small genome forward to prove the chunk budget is numerically
neutral.
"""

from __future__ import annotations

import numpy as np
import pytest

from tensorneat.common import TensorNEATError
from tensorneat.host_payload import _one_individual

#: The static node-row width of the hand-built payloads (``max_nodes``).
MAX_NODES = 8

#: The static connection-row width of the hand-built payloads (``max_conns``).
MAX_CONNS = 16

#: The attribute width of a node row; ``_one_individual`` validates ranks, not widths.
NODE_ATTR_WIDTH = 3

#: The key width of a connection row (``(in_key, out_key)``).
CONN_KEY_WIDTH = 2

#: The attribute width of a connection row (``weight`` at minimum).
CONN_ATTR_WIDTH = 1


def _payload(pop: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """A batch-first payload of ``pop`` individuals whose every value is distinct.

    Args:
        pop: The length of the population axis (``P``).

    Returns:
        The four arrays in payload order, shaped ``(P, max_nodes)``,
        ``(P, max_nodes, NODE_ATTR_WIDTH)``, ``(P, max_conns, CONN_KEY_WIDTH)``
        and ``(P, max_conns, CONN_ATTR_WIDTH)``. Values are a running counter per
        array, so individual ``p`` (and each field) is identifiable: a helper that
        returned a different individual or permuted the fields fails on values,
        not only on shapes.

    Notes:
        The widths are the documented static signature, so a message quoting
        ``(1, max_nodes)``/``(max_nodes,)`` is compared against real row counts.
    """
    return (
        np.arange(pop * MAX_NODES, dtype=np.int64).reshape(pop, MAX_NODES),
        np.arange(pop * MAX_NODES * NODE_ATTR_WIDTH, dtype=np.float64).reshape(
            pop, MAX_NODES, NODE_ATTR_WIDTH
        ),
        np.arange(pop * MAX_CONNS * CONN_KEY_WIDTH, dtype=np.int64).reshape(
            pop, MAX_CONNS, CONN_KEY_WIDTH
        ),
        np.arange(pop * MAX_CONNS * CONN_ATTR_WIDTH, dtype=np.float64).reshape(
            pop, MAX_CONNS, CONN_ATTR_WIDTH
        ),
    )


def test_a_one_individual_batch_is_returned_unchanged():
    """Invariant: a ``P == 1`` payload is accepted as-is and stays that individual.

    Oracle: the four arrays of :func:`_payload` at ``P = 1``, compared by value and
    by shape (and by dtype, so a promotion cannot quietly re-interpret keys).
    ``state.pop``/``best_genome`` at ``P = 1`` is exactly this spelling, which is
    the one ``Pipeline.show`` documents as native.

    Assertion: the returned tuple has four arrays of the input shapes
    ``(1, max_nodes)``/``(1, max_nodes, A)``/``(1, max_conns, key_width)``/
    ``(1, max_conns, W)``, each equal to its input array — no second population
    axis, no dropped individual.
    """
    batch = _payload(1)
    returned = _one_individual(batch)

    assert len(returned) == 4, (
        f"a payload is four arrays (node_keys, node_attrs, conn_keys, conn_attrs), "
        f"got {len(returned)} back"
    )
    for name, got, expected in zip(
        ("node_keys", "node_attrs", "conn_keys", "conn_attrs"), returned, batch
    ):
        assert got.shape == expected.shape, (
            f"{name} of a one-individual batch must come back unchanged with a "
            f"leading axis of 1, got {got.shape} for input {expected.shape}"
        )
        assert got.dtype == expected.dtype, (
            f"{name} must keep the payload's dtype, got {got.dtype} for input "
            f"{expected.dtype}"
        )
        np.testing.assert_array_equal(got, expected)


def test_the_bare_rows_are_promoted_to_the_same_one_individual_batch():
    """Invariant: the bare per-individual rows become that same ``P == 1`` batch.

    Oracle: strip the leading axis of a one-individual batch (what a checkpoint
    writes, one genome per file) and require ``_one_individual`` to hand back
    exactly the batch those rows came from — a value comparison, so a promotion
    that padded, sliced or reordered the rows fails.

    Assertion: the four promoted shapes are ``(1,)`` plus the bare row shapes, and
    each promoted array equals the corresponding row of :func:`_payload` at
    ``P = 1``.
    """
    batch = _payload(1)
    bare = tuple(np.asarray(field)[0] for field in batch)

    returned = _one_individual(bare)

    for name, got, expected in zip(
        ("node_keys", "node_attrs", "conn_keys", "conn_attrs"), returned, batch
    ):
        assert got.shape == expected.shape, (
            f"the bare {name} row must be promoted to the batch-first "
            f"{expected.shape}, got {got.shape}"
        )
        np.testing.assert_array_equal(got, expected)


@pytest.mark.parametrize("pop", [2, 3])
def test_a_multi_individual_payload_is_rejected_naming_the_two_spellings(pop):
    """Invariant: a payload with ``P > 1`` is a ``TensorNEATError`` naming both spellings.

    Oracle: the two spellings the helper documents — the one-individual batch
    ``(1, max_nodes)`` and the bare per-individual rows ``(max_nodes,)`` — plus the
    payload's own ``max_nodes``/``max_conns`` row widths, which are the static
    shape signature ``Pipeline`` builds against. A message that leaves out either
    spelling, or quotes no row width, does not tell the caller how to fix the call.

    Assertion: ``_one_individual`` raises ``TensorNEATError`` (not the
    "disagree about the population axis" message, which blames a different bug),
    the message names both accepted spellings and the offending population size,
    and no value comparison is needed because no array is returned at all.

    Notes:
        ``P = 2`` is the smallest population, ``P = 3`` proves the guard is not
        special-cased to two, and neither payload is silently sliced down to its
        first individual — which is exactly what this rejection prevents.
    """
    with pytest.raises(TensorNEATError) as excinfo:
        _one_individual(_payload(pop))

    message = str(excinfo.value)
    assert "(1, max_nodes)" in message, (
        "the rejection must name the one-individual batch spelling `(1, max_nodes)` "
        f"so the caller can slice the population; message was {message!r}"
    )
    assert "(max_nodes,)" in message, (
        "the rejection must name the bare per-individual row spelling "
        f"`(max_nodes,)`; message was {message!r}"
    )
    assert f"{pop} individuals" in message, (
        "the rejection must quote how many individuals the payload carries, so a "
        f"whole population `state.pop` is recognisable; message was {message!r}"
    )
    assert "disagree about the population axis" not in message, (
        "a P > 1 payload has one consistent population axis; reporting an axis "
        f"disagreement would misdiagnose it; message was {message!r}"
    )


def test_a_missing_payload_is_rejected_with_an_actionable_message():
    """Invariant: ``None`` raises ``TensorNEATError`` instead of a ``TypeError`` from ``list``.

    Oracle: ``Pipeline.auto_run`` documents ``best_genome`` as a payload *or*
    ``None`` when no generation produced a finite fitness, so ``None`` is a real
    caller mistake -- and its message must name both accepted spellings plus
    ``auto_run``/``best_genome``, the documented legitimate source.

    Assertion: ``TensorNEATError`` is raised, and the message contains
    ``(1, max_nodes)``, ``(max_nodes,)``, ``auto_run`` and ``best_genome``; a bare
    ``TypeError`` (what ``list(None)`` raises) would fail the exception type
    check alone.
    """
    with pytest.raises(TensorNEATError) as excinfo:
        _one_individual(None)

    message = str(excinfo.value)
    for token in ("(1, max_nodes)", "(max_nodes,)", "auto_run", "best_genome"):
        assert token in message, (
            f"the missing-payload message must mention {token!r}: it has to point "
            f"at a legitimate payload; message was {message!r}"
        )


def test_a_payload_that_is_not_four_arrays_is_rejected():
    """Invariant: a payload of the wrong arity keeps its own field-count message.

    Oracle: the documented payload is the four arrays ``(node_keys, node_attrs,
    conn_keys, conn_attrs)``; three arrays is malformed input and must be reported
    as such, never as a population-axis or multi-individual problem.

    Assertion: ``TensorNEATError`` naming the four field names and the count given.
    """
    with pytest.raises(TensorNEATError) as excinfo:
        _one_individual(_payload(1)[:3])

    message = str(excinfo.value)
    for name in ("node_keys", "node_attrs", "conn_keys", "conn_attrs"):
        assert name in message, f"the arity error must name {name}, got {message!r}"
    assert "3" in message, f"the arity error must quote the count given, got {message!r}"


def test_a_rank_zero_field_is_rejected():
    """Invariant: a scalar field is a rank error naming the field, not an axis error.

    Oracle: the payload's four arrays are all non-scalar; ``node_keys = 0.0`` is
    malformed input.

    Assertion: ``TensorNEATError`` naming field ``0`` (``node_keys``) and saying
    rank-0.
    """
    batch = _payload(1)
    with pytest.raises(TensorNEATError) as excinfo:
        _one_individual((np.float64(0.0),) + batch[1:])

    message = str(excinfo.value)
    assert "node_keys" in message and "rank-0" in message, (
        f"a scalar field must be reported as rank-0 node_keys, got {message!r}"
    )


@pytest.mark.parametrize(
    ("name", "index", "replacement"),
    [
        # node_keys above rank 2: (1, 1, max_nodes) -- a double population axis.
        ("node_keys", 0, lambda: np.zeros((1, 1, MAX_NODES), dtype=np.int64)),
        # conn_keys outside rank 2-3: a flat (max_conns,) row.
        ("conn_keys", 2, lambda: np.zeros(MAX_CONNS, dtype=np.int64)),
        # A key/attribute rank mismatch inside a batch-first payload.
        (
            "node_attrs",
            1,
            lambda: np.zeros((MAX_NODES, NODE_ATTR_WIDTH), dtype=np.float64),
        ),
    ],
)
def test_an_impossible_rank_is_rejected_naming_the_field(name, index, replacement):
    """Invariant: a field with a rank the spelling cannot have is reported by name.

    Oracle: the documented ranks — a batch-first payload is ``node_keys (P,
    max_nodes)``, ``node_attrs (P, max_nodes, A)``, ``conn_keys (P, max_conns,
    key_width)``, ``conn_attrs (P, max_conns, W)``; the bare spelling is one rank
    lower each. Each case swaps one field for a rank outside its spelling.

    Assertion: ``TensorNEATError`` naming the offending field and its achieved
    rank, i.e. the pre-existing rank validation still fires before the payload is
    normalised.
    """
    batch = list(_payload(1))
    batch[index] = replacement()

    with pytest.raises(TensorNEATError) as excinfo:
        _one_individual(tuple(batch))

    message = str(excinfo.value)
    assert name in message, (
        f"the rank error must name the offending field {name}, got {message!r}"
    )
    assert "rank-" in message, (
        f"the rank error must quote the achieved rank of {name}, got {message!r}"
    )


def test_arrays_that_disagree_about_the_population_axis_are_rejected():
    """Invariant: a ``P == 1`` node row beside a two-individual attribute row is rejected.

    Oracle: the batch has one shared population axis; ``node_attrs`` widened to
    two individuals cannot be paired with ``node_keys (1, max_nodes)`` at any
    interpretation, so no single individual exists to report.

    Assertion: ``TensorNEATError`` whose message lists the four shapes and says
    they disagree about the population axis (the message that survives *only* for
    a genuine disagreement, now that ``P > 1`` gets its own rejection).
    """
    batch = list(_payload(1))
    batch[1] = np.zeros((2, MAX_NODES, NODE_ATTR_WIDTH), dtype=np.float64)

    with pytest.raises(TensorNEATError) as excinfo:
        _one_individual(tuple(batch))

    message = str(excinfo.value)
    assert "disagree about the population axis" in message, (
        "a genuine axis disagreement must keep the documented population-axis "
        f"message; got {message!r}"
    )
    assert "(1," in message and "(2," in message, (
        f"that message must quote the four shapes, got {message!r}"
    )


def test_run_network_chunks_the_tiled_batch(monkeypatch):
    """Invariant: a tiled batch wider than the budget runs in row-contiguous individual groups.

    Oracle: the unchunked path itself. The same ``P * S = 4 * 3`` batch is
    evaluated once with a budget above the tiled width and once with the budget
    monkeypatched down to ``8`` rows; the concatenated outputs must be
    bit-identical, which is the claim behind the regrouping (every forward op is
    row-local, and the one population-wide quantity — the level bound — is a
    maximum). The two caught errors this test exists for are a chunk that is not
    a multiple of ``S`` (its rows would belong to different individuals) and a
    concatenation in the wrong order.

    Assertion: with ``EVAL_CHUNK_ROWS = 8`` and ``S = 3`` the budget resolves to
    two individuals per chunk, so ``prepare`` is called twice on a
    two-individual slice and the forward sees input blocks of six rows each
    (``[0:6]`` then ``[6:12]``); the unchunked pass prepares the whole population
    once and sees all twelve rows; the two outputs compare equal
    (``equal_nan=True``).

    Notes:
        ``EVAL_CHUNK_ROWS`` is a module constant read inside
        ``_prepared_query`` at trace time, so the monkeypatch has to wrap the
        *build*, which is why the two runs are separate ``support.runner`` calls.
        The payload's individuals carry different edge weights, so an output
        concatenated in the wrong row order cannot compare equal by symmetry.
    """
    import numpy as np

    import tensorneat.genome as genome_pkg
    import tensorneat.host_payload as host_payload
    from support.payloads import chain_payload, default_genome, genome_state
    from support.runner import run_defn

    pop_size, samples = 4, 3
    payload = chain_payload(pop_size, 8, 16)
    edges = [(0, 2), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7)]
    for p in range(pop_size):  # a different weight scale per individual
        for row, (source, target) in enumerate(edges):
            payloads_row = 1.0 + 0.25 * p
            payload.conn_attrs[p, row, 0] = np.float32(payloads_row)
    genome_object = default_genome(max_nodes=8, max_conns=16, max_in_degree=4)
    state = genome_state(genome_object)
    transformed = _derive_schedule(genome_object, state, payload)
    inputs = np.arange(pop_size * samples * 2, dtype=np.float32).reshape(
        pop_size * samples, 2
    )
    inputs = (inputs * 0.1 - 0.3).astype(np.float32)
    width = len(transformed)

    calls = []

    def prepare(state_arg, schedule, group_payload, factor):
        """Record the group the chunk plan prepared, then prepare it for real."""
        calls.append(("prepare", int(group_payload.node_keys.shape[0]), int(factor)))
        return genome_object.prepare_pass(
            state_arg,
            schedule,
            group_payload.node_keys,
            group_payload.node_attrs,
            group_payload.conn_keys,
            group_payload.conn_attrs,
            factor,
        )

    real_forward = genome_object.forward_prepared

    def forward(state_arg, schedule, prepared, query_inputs):
        """Record how many rows the group forward was queried with."""
        calls.append(("forward", int(query_inputs.shape[0])))
        return real_forward(state_arg, schedule, prepared, query_inputs)

    def traced(*arrays):
        """Prepare the tiled pass and run its one query, inside one trace."""
        schedule = _rebuild_transformed(arrays[:width])
        batch = genome_pkg.GenomeTensors(*arrays[width : width + 4])
        query = host_payload._prepared_query(
            state,
            forward,
            prepare,
            schedule,
            batch,
            pop_size * samples,
        )
        return query(arrays[-1])

    args = (*tuple(transformed), *payload.args(), inputs)

    with monkeypatch.context() as patch:
        patch.setattr(host_payload, "EVAL_CHUNK_ROWS", 8)
        chunked = run_defn(traced, *args)

    chunked_calls = list(calls)
    calls.clear()

    with monkeypatch.context() as patch:
        patch.setattr(host_payload, "EVAL_CHUNK_ROWS", 10**9)
        unchunked = run_defn(traced, *args)

    unchunked_calls = list(calls)

    assert chunked_calls == [
        ("prepare", 2, samples),
        ("prepare", 2, samples),
        ("forward", 2 * samples),
        ("forward", 2 * samples),
    ], (
        "a budget of 8 rows at S=3 must prepare two individuals per group and "
        f"query blocks of six rows; got {chunked_calls}"
    )
    assert unchunked_calls == [
        ("prepare", pop_size, samples),
        ("forward", pop_size * samples),
    ], (
        "a budget above the tiled width must keep the whole population in one "
        f"pass; got {unchunked_calls}"
    )
    assert chunked.shape == unchunked.shape == (pop_size * samples, 1)
    assert np.array_equal(chunked, unchunked, equal_nan=True), (
        "the chunked and unchunked forwards disagree: "
        f"{chunked.ravel()} vs {unchunked.ravel()}"
    )


def _derive_schedule(genome_object, state, payload):
    """Derive ``payload``'s host execution schedule through the public entry point.

    Args:
        genome_object: The genome whose static shapes match ``payload``.
        state: The host ``State`` returned by ``payloads.genome_state``.
        payload: A ``support.payloads.Payload`` batch.

    Returns:
        The ``common.Transformed`` namedtuple with host numpy leaves.
    """
    from support.runner import run_defn

    return run_defn(
        lambda node_keys, conn_keys: genome_object.transform(
            state, node_keys, conn_keys
        ),
        payload.node_keys,
        payload.conn_keys,
    )


def _rebuild_transformed(fields):
    """Rebuild the schedule namedtuple some host fields came from.

    Args:
        fields: The leading ``len(Transformed._fields)`` arrays of the traced call.

    Returns:
        A ``common.Transformed`` whose leaves are the traced tensors.
    """
    import tensorneat.common as common

    return common.Transformed(*fields)
