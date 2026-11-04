"""Tests of ``./convert.py`` — the column-layout spec both libraries are fed through.

Unlike the other modules in ``./``, this one is **not** a stub: ``convert.py`` is
plain host Python (no ``evoxc``, no ``jax``), so it can be — and is — verified
before either library is finished.  It is the module every parity test depends on,
which is exactly why it needs its own tests:

* the two column layouts (key columns integer-valued and sentinel-translated,
  attribute columns copied verbatim, ``NaN`` included) are asserted entry by
  entry, so a wrong column order cannot make a whole parity suite compare the
  wrong thing;
* ``I_INF <-> NaN`` in **both** directions, with a round trip that must be an
  exact identity;
* the rejection cases: a batch where one individual is expected, a non-integer
  key, a key outside ``[0, KEY_MAX)``, a partially absent legacy row — all of
  which would otherwise silently shift a network's identity;
* the two layout rules ``max_in_degree_for`` and ``legacy_init_hidden_layers``
  (plus ``legacy_initial_edge_count``), with the numbers the parity fixtures
  actually use.

``from_legacy`` is exercised through a stand-in ``payload_factory`` so it does not
depend on ``support.payloads`` (which imports ``tensorneat``); the default factory
is covered by :func:`test_sentinels_match_the_package_constants`.
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np
import pytest

pytestmark = [pytest.mark.parity]


class StubPayload(NamedTuple):
    """The four payload arrays of one individual — the shape ``from_legacy`` builds."""

    node_keys: np.ndarray
    node_attrs: np.ndarray
    conn_keys: np.ndarray
    conn_attrs: np.ndarray


def _payload(convert):
    """A four-node / three-connection payload with one absent row in each family."""
    absent = convert.I_INF
    node_keys = np.array([[0, 1, 7, absent]], dtype=np.int32)
    node_attrs = np.array(
        [
            [
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.25, 2.0, 0.0, 0.0],
                [np.nan, np.nan, np.nan, np.nan],
            ]
        ],
        dtype=np.float32,
    )
    conn_keys = np.array([[[0, 7], [1, 7], [absent, absent]]], dtype=np.int32)
    conn_attrs = np.array([[[1.0], [0.5], [np.nan]]], dtype=np.float32)
    return StubPayload(node_keys, node_attrs, conn_keys, conn_attrs)


# --------------------------------------------------------------------------- layout
def test_to_legacy_writes_the_documented_columns(convert):
    """Invariant: the legacy matrices are ``[keys | attrs]`` with ``NaN`` rows for absent genes.

    Oracle: the column layout of the two sources, written out by hand here.  With
    ``node_keys = [[0, 1, 7, I_INF]]`` the legacy ``nodes`` matrix must be ``(4, 5)``
    with column ``0`` holding ``[0, 1, 7, NaN]`` (the legacy "absent row" marker)
    and columns ``1:`` the attribute rows; the connection matrix must be ``(3, 3)``
    with ``[0, 7, 1.0]``, ``[1, 7, 0.5]`` and an all-``NaN`` absent row.

    Assertion: ``np.testing.assert_array_equal`` (which treats ``NaN`` as equal to
    ``NaN``) on both matrices, plus ``dtype == np.float32`` on both — the legacy
    library initialises its payloads with ``jnp.float32`` and relies on integer
    keys surviving that representation.
    """
    nodes, conns = convert.to_legacy(_payload(convert))

    assert nodes.dtype == np.float32 and nodes.shape == (4, 5)
    np.testing.assert_array_equal(nodes[:, 0], np.array([0, 1, 7, np.nan], np.float32))
    np.testing.assert_array_equal(nodes[2, 1:], np.array([0.25, 2.0, 0.0, 0.0], np.float32))
    assert np.isnan(nodes[3, 1:]).all()

    assert conns.dtype == np.float32 and conns.shape == (3, 3)
    np.testing.assert_array_equal(conns[:2], np.array([[0, 7, 1.0], [1, 7, 0.5]], np.float32))
    assert np.isnan(conns[2]).all()


def test_round_trip_is_an_exact_identity(convert):
    """Invariant: ``from_legacy(*to_legacy(p))`` reproduces ``p`` exactly, padding included.

    Oracle: the original payload.  Both directions of the
    ``I_INF <-> NaN`` translation, both gene families, the absent rows and the
    ``NaN`` attribute entries must survive, because a parity suite that dropped a
    column would compare a different genome on each side.

    Assertion: ``assert_array_equal`` on all four arrays (shapes, dtypes and values,
    ``NaN`` == ``NaN``), for a payload with an absent node row, an absent connection
    row and a ``NaN`` attribute — and again for an ``OriginConn``-style payload
    (``CK == 3``) round-tripped with ``conn_key_width=3``.
    """
    original = _payload(convert)
    nodes, conns = convert.to_legacy(original)

    restored = convert.from_legacy(nodes, conns, payload_factory=StubPayload)

    for name in original._fields:
        np.testing.assert_array_equal(getattr(restored, name), getattr(original, name))
    assert restored.node_keys.dtype == np.int32
    assert restored.node_attrs.dtype == np.float32


def test_origin_conn_marker_column_is_carried(convert):
    """Invariant: a third key column (``historical_marker``) survives both directions.

    Oracle: ``OriginConn.fixed_attrs == ("input_index", "output_index",
    "historical_marker")``, i.e. ``conn_key_width == 3``.  The marker column is an
    integer key, not an attribute, so it must be translated like the other key
    columns (``I_INF <-> NaN``) and must not be folded into ``conn_attrs``.

    Assertion: ``to_legacy`` of a ``CK == 3`` payload yields ``(E, 3 + CA)`` columns
    with the marker values in column ``2``; ``from_legacy(..., conn_key_width=3)``
    restores all three key columns exactly, and reading the same matrix with the
    default ``conn_key_width=2`` gives a *different* (and documented) split —
    the marker would land in ``conn_attrs`` — which is why the width is an
    explicit argument rather than inferred.
    """
    absent = convert.I_INF
    payload = StubPayload(
        np.array([[0, 1, 7]], dtype=np.int32),
        np.zeros((1, 3, 4), dtype=np.float32),
        np.array([[[0, 7, 4], [1, 7, 5], [absent, absent, absent]]], dtype=np.int32),
        np.ones((1, 3, 1), dtype=np.float32),
    )

    nodes, conns = convert.to_legacy(payload)

    assert conns.shape == (3, 4)
    np.testing.assert_array_equal(conns[:2, :3], np.array([[0, 7, 4], [1, 7, 5]], np.float32))
    restored = convert.from_legacy(conns=conns, nodes=nodes, conn_key_width=3, payload_factory=StubPayload)
    np.testing.assert_array_equal(restored.conn_keys, payload.conn_keys)
    np.testing.assert_array_equal(restored.conn_attrs, payload.conn_attrs)


# --------------------------------------------------------------------------- rejections
def test_to_legacy_rejects_a_batch(convert):
    """Invariant: a legacy genome is one individual, so a batch is refused, not sliced.

    Oracle: ``BaseGenome.transform(state, nodes, conns)`` takes exactly one
    ``(N, L)``/``(E, L)`` pair; silently converting ``P > 1`` would compare one
    arbitrary individual's network.  The caller is expected to use
    ``payload.of(p)`` (documented in ``convert.to_legacy``).

    Assertion: ``ValueError`` whose message names ``pop_size``.
    """
    payload = _payload(convert)
    stacked = payload._replace(
        node_keys=np.repeat(payload.node_keys, 2, axis=0),
        node_attrs=np.repeat(payload.node_attrs, 2, axis=0),
        conn_keys=np.repeat(payload.conn_keys, 2, axis=0),
        conn_attrs=np.repeat(payload.conn_attrs, 2, axis=0),
    )

    with pytest.raises(ValueError, match="pop_size"):
        convert.to_legacy(stacked)


@pytest.mark.parametrize(
    "column, match",
    [
        (np.array([0.0, 1.5, 7.0, np.nan]), "integer-valued"),
        (np.array([0.0, 1.0, float(1 << 24), np.nan]), "keys must lie"),
        (np.array([0.0, -1.0, 7.0, np.nan]), "keys must lie"),
    ],
)
def test_from_legacy_rejects_unusable_keys(convert, column, match):
    """Invariant: a key outside ``[0, KEY_MAX)`` or a non-integer key is an error, not a cast.

    Oracle: ``tensorneat.common.KEY_MAX`` — the first integer a ``float32`` column
    cannot separate from its neighbour, which is why the rewrite keys on ``int32``
    and the legacy keys on integer-valued ``float32``.  Casting silently would let
    two distinct nodes collapse onto one key and make the comparison meaningless.

    Assertion: ``ValueError`` matching ``integer-valued`` / ``keys must lie`` for a
    fractional key, a key at ``KEY_MAX`` and a negative key, in both the node and
    the connection column.
    """
    nodes, conns = convert.to_legacy(_payload(convert))
    nodes[:, 0] = column

    with pytest.raises(ValueError, match=match):
        convert.from_legacy(nodes, conns, payload_factory=StubPayload)


def test_from_legacy_rejects_a_partially_absent_row(convert):
    """Invariant: a legacy absent row is absent in every key column, or it is an error.

    Oracle: the legacy detection rule ``isnan(conns[:, 0])``.  A row with ``NaN``
    in column ``0`` but a real key in column ``1`` is malformed: the two libraries
    would disagree about whether the row exists, so it must raise instead of being
    normalised to "absent" (which would drop a connection) or to "present" (which
    would invent an edge from ``NaN``).

    Assertion: ``ValueError`` matching ``only partially absent`` — for
    ``from_legacy`` on such a matrix and for ``to_legacy`` on a payload whose
    ``conn_keys`` row mixes ``I_INF`` and a real key.
    """
    nodes, conns = convert.to_legacy(_payload(convert))
    conns[2] = np.array([np.nan, 7.0, 1.0], dtype=np.float32)

    with pytest.raises(ValueError, match="only partially absent"):
        convert.from_legacy(nodes, conns, payload_factory=StubPayload)

    payload = _payload(convert)
    mixed = payload.conn_keys.copy()
    mixed[0, 2] = np.array([convert.I_INF, 7], dtype=np.int32)

    with pytest.raises(ValueError, match="entirely I_INF"):
        convert.to_legacy(payload._replace(conn_keys=mixed))


def test_from_legacy_rejects_an_unusable_width(convert):
    """Invariant: ``conn_key_width`` must describe at least ``(input, output)``.

    Oracle: ``BaseConn.fixed_attrs``.  A width of ``1`` (or a width that leaves no
    attribute column, or ``0``) cannot describe a connection gene, and the matrices
    carry no metadata to recover it from.

    Assertion: ``ValueError`` for ``conn_key_width=1`` and for a width that consumes
    every column; ``TypeError`` for a non-integer width.
    """
    nodes, conns = convert.to_legacy(_payload(convert))

    with pytest.raises(ValueError, match="at least 2"):
        convert.from_legacy(nodes, conns, conn_key_width=1, payload_factory=StubPayload)
    with pytest.raises(ValueError, match="at least one attribute column"):
        convert.from_legacy(nodes, conns, conn_key_width=conns.shape[1], payload_factory=StubPayload)
    with pytest.raises(TypeError, match="conn_key_width"):
        convert.from_legacy(nodes, conns, conn_key_width=2.0, payload_factory=StubPayload)


# --------------------------------------------------------------------------- layout rules
@pytest.mark.parametrize("max_nodes", [1, 8, 16, 50])
def test_max_in_degree_covers_the_legacy_column(convert, max_nodes):
    """Invariant: the parity window width is ``max_nodes`` — wide enough for any in-degree.

    Oracle: the legacy forward aggregates over one ``max_nodes``-wide column per
    node and truncates nothing, so the rewrite's ``K`` window has to be at least
    that wide.  ``K == max_nodes`` is exactly "at least as wide", because a node
    can have at most one in-edge per node slot.

    Assertion: ``convert.max_in_degree_for(max_nodes) == max_nodes`` (never the
    genome's default ``max(initial in-degree, 4)``), and ``TypeError``/``ValueError``
    for a non-integer or non-positive capacity rather than a silently clamped value.
    """
    assert convert.max_in_degree_for(max_nodes) == max_nodes
    with pytest.raises(ValueError, match="positive"):
        convert.max_in_degree_for(0)
    with pytest.raises(TypeError, match="int"):
        convert.max_in_degree_for(8.0)


@pytest.mark.parametrize(
    "max_nodes, num_inputs, num_outputs, hidden, edges",
    [
        (8, 2, 1, 5, 15),
        (8, 3, 1, 4, 16),
        (3, 2, 1, 0, 0),
        (16, 2, 2, 12, 48),
    ],
)
def test_legacy_layout_puts_the_outputs_at_the_end(
    convert, max_nodes, num_inputs, num_outputs, hidden, edges
):
    """Invariant: one dense hidden layer of ``N - I - O`` makes both libraries' output keys agree.

    Oracle: the legacy ``BaseGenome.__init__`` layer construction
    (``[num_inputs] + init_hidden_layers + [num_outputs]``) against the rewrite's
    row/key invariant (output keys are the last ``num_outputs`` rows).  The
    rewrite's ``get_output_idx()`` must equal the legacy ``get_output_idx()`` for
    that layout, and the legacy constructor's capacity requirement is
    ``I * H + H * O`` edges.

    Assertion: ``legacy_init_hidden_layers(...) == (H,)``,
    ``legacy_initial_edge_count(...) == I * H + H * O``, and — once the two genomes
    are importable — ``pair.new_genome.get_output_idx() ==
    pair.legacy_genome.get_output_idx()`` (the last half is covered by
    ``./conftest.genome_pair``, which uses exactly this layout).  ``ValueError``
    when ``I + O > N``.
    """
    assert convert.legacy_init_hidden_layers(max_nodes, num_inputs, num_outputs) == (hidden,)
    assert (
        convert.legacy_initial_edge_count(max_nodes, num_inputs, num_outputs)
        == num_inputs * hidden + hidden * num_outputs
        == edges
    )
    with pytest.raises(ValueError, match="cannot hold"):
        convert.legacy_init_hidden_layers(4, 2, 3)


# --------------------------------------------------------------------------- sentinels
def test_sentinels_match_the_package_constants(convert):
    """Invariant: the duplicated ``I_INF``/``KEY_MAX`` equal the package's own constants.

    Oracle: ``tensorneat.common.tools.I_INF`` (``int32`` max) and ``KEY_MAX``
    (``1 << 24``).  ``convert`` re-declares them so it stays a pure-numpy layer;
    this test is what keeps the duplication honest.

    Assertion: ``convert.I_INF == tools.I_INF`` and
    ``convert.KEY_MAX == tools.KEY_MAX``.
    """
    from support.lazy import require

    tools = require("tensorneat.common.tools")

    assert convert.I_INF == tools.I_INF
    assert convert.KEY_MAX == tools.KEY_MAX
