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

No backend, no ``Pipeline`` build, no ``jax``/``torch``: the helper is
host-only, so these tests are plain numpy.
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
