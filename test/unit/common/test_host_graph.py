"""Unit tests for ``tensorneat.common.host_graph`` — host-side graph algorithms.

What this module proves
-----------------------
``host_graph`` is the host-side counterpart of the in-graph schedule: it answers
*what does one genome look like* (topological order, depths, reachability,
"does this mutation close a cycle?"), works on plain ``int`` keys and Python
containers, raises Python exceptions, and never participates in a trace. It has
exactly three consumers (``genome.network_dict``, ``genome.sympy_func``,
``genome.visualize``), so the tests assert the shapes those consumers rely on:
both return values of :func:`~tensorneat.common.host_graph.topological_sort`,
the exact fixpoint of :func:`~tensorneat.common.host_graph.find_useful_nodes`,
and the precise error type of every failure mode.

Oracles: hand enumeration. Every fixture below is a small genome written out
literally (nodes and ``(in, out)`` pairs), and every expectation — the layer
partition, the depth of each key, the membership of the useful set — is written
down literally next to it. No test calls a second graph implementation to check
the first; where an independent walk exists (``node_depth`` vs the schedule's
``level``) the agreement is asserted in ``test_tensorization.py``, which owns the
schedule side.

Gotchas this module encodes
---------------------------
* A cycle is an **error** here (``GraphError``) and *data* in the schedule
  (``level == I_INF``); ``check_cycles`` answers "would this edge close a cycle",
  not "is this graph acyclic", and returns ``True`` for a self-loop while
  returning ``False`` for an edge that already exists in an acyclic graph.
* Determinism does not come from set iteration order: nodes freed in the same
  round are emitted in ascending key order, so the same genome gives the same
  layers regardless of how the ``set`` was built.
* Nothing here takes a batch axis or a sentinel: a caller with a ``(P, max_nodes,
  ...)`` payload selects one member first, and this module only sees ``int`` keys.
  :func:`~tensorneat.common.host_graph.decode_graph` is the single exception —
  it is the one padding crossing point, and it accepts both the ``float32``/
  ``NaN`` spelling of :func:`gene_rows` and the canonical ``int32``/``I_INF``
  one.

Backend-independent (no ``evoxc`` op is called); no ``jax``/``torch``, no network.
"""

from __future__ import annotations

import importlib

import numpy as np
import pytest

#: ``int32`` max — the key sentinel of the canonical split payload. A literal
#: here (not an import) so this module never depends on ``common.tools``'
#: ``evoxc`` import just to spell the sentinel.
I_INF = 2147483647


@pytest.fixture
def host_graph(common):
    """The module under test: ``tensorneat.common.host_graph``.

    Imported by name because the package ``__init__`` does not re-export it (it
    is a leaf only ``genome`` uses). The ``common`` fixture runs first, so the
    skip that covers the unimplemented ``src/`` skeleton is reported once, with
    its own reason.
    """
    return importlib.import_module("tensorneat.common.host_graph")


@pytest.fixture
def errors(common):
    """The error hierarchy of the package: ``tensorneat.common.errors``.

    Imported by name (not read out of ``host_graph``'s namespace) so the
    expectation is the *public* class, not a name the module happens to
    re-export: ``GraphError`` is a broken topology (duplicate key, dangling
    endpoint, cycle, unreachable node) and ``ConfigError`` is a caller mistake
    (bad fixed widths, unknown input/output/candidate keys). ``common`` is
    requested first so the skeleton skip is reported once, with its own reason.
    """
    return importlib.import_module("tensorneat.common.errors")


@pytest.fixture
def feedforward():
    """A hand-written 6-node feedforward genome: inputs ``0, 1``; output ``4``.

    Connections, literally: ``0→2``, ``1→2``, ``2→3``, ``3→4`` and a dangling
    ``1→5`` that reaches no output. Hence the layer partition
    ``[[0, 1], [2, 5], [3], [4]]``, the depths
    ``{0: 0, 1: 0, 2: 1, 3: 2, 4: 3, 5: 1}`` and the useful set
    ``{0, 1, 2, 3, 4}`` (``5`` is excluded).
    """
    nodes = {0, 1, 2, 3, 4, 5}
    conns = {(0, 2), (1, 2), (2, 3), (3, 4), (1, 5)}
    return nodes, conns


@pytest.fixture
def cyclic():
    """A hand-written 3-node cyclic genome: ``0→1``, ``1→2``, ``2→1``."""
    nodes = {0, 1, 2}
    conns = {(0, 1), (1, 2), (2, 1)}
    return nodes, conns


@pytest.fixture
def gene_rows():
    """One genome's float payload rows: ``(nodes, conns)`` with ``NaN`` padding.

    Shape ``(max_nodes, 2)`` for nodes (key + one attribute) and
    ``(max_conns, 3)`` for connections (``in_key``, ``out_key``, weight), padded
    with ``NaN`` so :func:`~tensorneat.common.host_graph.decode_graph` has to skip
    the unused rows — the only place in the package where a float/``NaN`` payload
    still exists.
    """
    nan = float("nan")
    nodes = np.array(
        [[0.0, 0.0], [1.0, 0.0], [2.0, 0.5], [3.0, 0.25], [nan, nan]], dtype=np.float32
    )
    conns = np.array(
        [
            [0.0, 2.0, 0.5],
            [1.0, 2.0, -0.5],
            [2.0, 3.0, 1.0],
            [nan, nan, nan],
        ],
        dtype=np.float32,
    )
    return nodes, conns


# --------------------------------------------------------------------------- #
# decode_graph
# --------------------------------------------------------------------------- #


def test_decode_graph_skips_nan_padding_and_casts_keys(host_graph, gene_rows):
    """Invariant: ``{int(key): attrs}`` for the used rows and a set of ``(in_key, out_key)``.

    Oracle: hand enumeration of :func:`gene_rows` — four used node rows (keys
    ``0..3``, attributes starting *after* the one fixed column) and three used
    connection rows, with the ``NaN`` rows dropped. The insertion order must
    follow the payload row order.
    """
    nodes, conns = gene_rows

    nodes_out, conns_out = host_graph.decode_graph(nodes, conns)

    assert isinstance(nodes_out, dict)
    assert isinstance(conns_out, set)

    # Attributes are ``row[node_attrs:]`` with the default ``node_attrs=1``.
    assert nodes_out == {0: (0.0,), 1: (0.0,), 2: (0.5,), 3: (0.25,)}
    assert list(nodes_out) == [0, 1, 2, 3]  # payload row order, NaN row skipped
    assert all(type(key) is int for key in nodes_out)
    assert all(type(value) is tuple for value in nodes_out.values())
    assert all(type(item) is float for value in nodes_out.values() for item in value)

    assert conns_out == {(0, 2), (1, 2), (2, 3)}
    assert all(type(pair) is tuple for pair in conns_out)
    assert all(type(key) is int for pair in conns_out for key in pair)

    # A wider node row drops exactly the declared fixed prefix (``node_attrs=2``
    # keeps the third column), which is the same rule the network dict reads.
    wide_nodes = np.array(
        [[0.0, 7.0, 0.25], [1.0, 8.0, 0.75]], dtype=np.float32
    )
    wide_out, _ = host_graph.decode_graph(
        wide_nodes, np.empty((0, 3), dtype=np.float32), node_attrs=2
    )
    assert wide_out == {0: (0.25,), 1: (0.75,)}

    # The canonical ``int32``/``I_INF`` spelling decodes to the same result: the
    # padding test is "NaN *or* I_INF", never a dtype branch.
    int_nodes = np.array([[0, 0], [1, 0], [I_INF, 0]], dtype=np.int32)
    int_conns = np.array([[0, 1, 0], [I_INF, I_INF, 0]], dtype=np.int32)
    int_nodes_out, int_conns_out = host_graph.decode_graph(int_nodes, int_conns)
    assert int_nodes_out == {0: (0.0,), 1: (0.0,)}
    assert int_conns_out == {(0, 1)}


def test_decode_graph_collapses_duplicate_connection_pairs(host_graph):
    """Invariant: the same ``(in_key, out_key)`` twice yields one element of the returned set.

    Oracle: the documented "a set makes the topology"; the schedule, not this
    module, decides whether a duplicate is legal — so the decoded set must simply
    be smaller than the row count.
    """
    nodes = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]], dtype=np.float32)
    conns = np.array(
        [[0.0, 2.0, 0.5], [0.0, 2.0, -0.75], [1.0, 2.0, 0.25]], dtype=np.float32
    )

    _, conns_out = host_graph.decode_graph(nodes, conns)

    assert conns_out == {(0, 2), (1, 2)}
    assert len(conns_out) == 2 < len(conns)


def test_decode_graph_rejects_a_duplicate_node_key(host_graph, gene_rows, errors):
    """Invariant: two used node rows with the same key raise ``GraphError``.

    Oracle: the documented ``Raises`` clause; a duplicate key makes the
    key→attribute map lossy, so it is a genome-structure error and not a warning.
    """
    nodes, conns = gene_rows
    duplicate_nodes = np.array(
        [[0.0, 0.0], [1.0, 0.0], [1.0, 0.5], [np.nan, np.nan]], dtype=np.float32
    )

    with pytest.raises(errors.GraphError) as excinfo:
        host_graph.decode_graph(duplicate_nodes, conns)
    assert "1" in str(excinfo.value)

    # The same payload with the duplicate key renamed decodes without complaint,
    # so the failure really is the repeated key and not the payload shape.
    # (``conns[:2]`` keeps the endpoint set inside the used node rows.)
    renamed = duplicate_nodes.copy()
    renamed[2, 0] = 2.0
    nodes_out, _ = host_graph.decode_graph(renamed, conns[:2])
    assert sorted(nodes_out) == [0, 1, 2]


def test_decode_graph_rejects_a_connection_to_an_absent_node(host_graph, gene_rows, errors):
    """Invariant: an endpoint that names no used node row raises ``GraphError``.

    Oracle: the documented clause; the payload is truncated so one endpoint key
    has no node row, which is what a truncated connection list looks like.
    """
    nodes, conns = gene_rows
    # Keep the node rows for keys 0, 1, 2 only; the connection ``2→3`` then
    # names an output key with no node row.
    truncated_nodes = nodes[:3]

    with pytest.raises(errors.GraphError) as excinfo:
        host_graph.decode_graph(truncated_nodes, conns)
    assert "3" in str(excinfo.value)

    # An input-side hole is the same error (the source endpoint is validated too).
    hole_in_nodes = np.array([[1.0, 0.0], [2.0, 0.0], [3.0, 0.0]], dtype=np.float32)
    with pytest.raises(errors.GraphError):
        host_graph.decode_graph(hole_in_nodes, conns)

    # A *used* row whose destination column is padding cannot be keyed at all:
    # documented as ``GraphError`` because the topology is broken, not the config.
    nan = float("nan")
    padding_destination = np.array(
        [[0.0, 2.0, 0.5], [0.0, nan, 1.0]], dtype=np.float32
    )
    with pytest.raises(errors.GraphError):
        host_graph.decode_graph(nodes, padding_destination)


def test_decode_graph_rejects_a_bad_fixed_prefix(host_graph, gene_rows, errors):
    """Invariant: ``node_attrs < 1`` / ``conn_attrs < 2`` and a too-short row raise ``ConfigError``.

    Oracle: the documented widths (``fixed_attrs == ["idx"]`` for nodes,
    ``["input_index", "output_index"]`` for connections); a row shorter than its
    fixed prefix cannot even be keyed, so it is a configuration error surfaced at
    export time.
    """
    nodes, conns = gene_rows
    empty_conns = np.empty((0, 3), dtype=np.float32)

    with pytest.raises(errors.ConfigError):
        host_graph.decode_graph(nodes, conns, node_attrs=0)

    with pytest.raises(errors.ConfigError):
        host_graph.decode_graph(nodes, conns, node_attrs=-1)

    with pytest.raises(errors.ConfigError):
        host_graph.decode_graph(nodes, conns, conn_attrs=1)

    with pytest.raises(errors.ConfigError):
        host_graph.decode_graph(nodes, conns, conn_attrs=0)

    # A node row shorter than its declared fixed prefix.
    short_node_rows = np.array([[0.0]], dtype=np.float32)
    with pytest.raises(errors.ConfigError):
        host_graph.decode_graph(short_node_rows, empty_conns, node_attrs=2)

    # A node row with no columns at all cannot hold the key.
    zero_width_node_rows = np.empty((1, 0), dtype=np.float32)
    with pytest.raises(errors.ConfigError):
        host_graph.decode_graph(zero_width_node_rows, empty_conns)

    # A connection row shorter than ``conn_attrs`` (here: missing the weight).
    two_used_nodes = np.array([[0.0, 0.0], [2.0, 0.0]], dtype=np.float32)
    short_conn_rows = np.array([[0.0, 2.0]], dtype=np.float32)
    with pytest.raises(errors.ConfigError):
        host_graph.decode_graph(two_used_nodes, short_conn_rows, conn_attrs=3)

    # A connection row with no columns at all cannot hold both endpoints.
    zero_width_conn_rows = np.empty((1, 0), dtype=np.float32)
    with pytest.raises(errors.ConfigError):
        host_graph.decode_graph(two_used_nodes, zero_width_conn_rows)

    # The widths the contract declares (1 and 2) are accepted on the same payload,
    # so each ``ConfigError`` above is the declared bound and not a blanket reject.
    _, _ = host_graph.decode_graph(nodes, conns, node_attrs=1, conn_attrs=2)


# --------------------------------------------------------------------------- #
# topological_sort
# --------------------------------------------------------------------------- #


def test_topological_sort_returns_layers_and_their_concatenation(host_graph, feedforward):
    """Invariant: ``topo_layers`` is the literal ``[[0, 1], [2, 5], [3], [4]]``.

    Oracle: hand enumeration of :func:`feedforward`; each layer is sorted
    ascending, layer ``0`` is the in-degree-0 set (the inputs), ``topo_order`` is
    their concatenation and every node appears exactly once across the partition.
    """
    nodes, conns = feedforward

    topo_order, topo_layers = host_graph.topological_sort(nodes, conns)

    assert topo_layers == [[0, 1], [2, 5], [3], [4]]
    assert topo_order == [0, 1, 2, 5, 3, 4]
    assert topo_order == [node for layer in topo_layers for node in layer]
    assert sorted(topo_order) == sorted(nodes)
    assert len(topo_order) == len(set(topo_order)) == len(nodes)

    # Each node sits strictly below all of its predecessors: the literal depths
    # of the fixture, written out independently of the returned layers.
    layer_of = {
        node: index for index, layer in enumerate(topo_layers) for node in layer
    }
    for in_key, out_key in conns:
        assert layer_of[out_key] > layer_of[in_key]


def test_topological_sort_is_deterministic_regardless_of_input_order(host_graph, feedforward):
    """Invariant: the same topology given as lists in different orders yields the same layers.

    Oracle: the ascending-within-layer rule of the docstring, checked by feeding
    the reversed node list and the reversed connection list as well — set
    iteration order must not leak into the result, because visualisation and
    formula output are diffed between runs.
    """
    nodes, conns = feedforward
    expected_layers = [[0, 1], [2, 5], [3], [4]]
    expected_order = [0, 1, 2, 5, 3, 4]

    variants = [
        (nodes, conns),
        (list(nodes), list(conns)),
        (sorted(nodes), sorted(conns)),
        (sorted(nodes, reverse=True), sorted(conns, reverse=True)),
        (list(reversed(sorted(nodes))), list(reversed(sorted(conns)))),
    ]
    for variant_nodes, variant_conns in variants:
        topo_order, topo_layers = host_graph.topological_sort(
            variant_nodes, variant_conns
        )
        assert topo_layers == expected_layers
        assert topo_order == expected_order


def test_topological_sort_does_not_mutate_its_arguments(host_graph, feedforward):
    """Invariant: ``nodes`` and ``conns`` are unchanged after the call.

    Oracle: a copy taken before, compared after; the pre-rewrite version copied
    its arguments, and a caller that reuses its genome between rounds would
    otherwise lose edges.
    """
    nodes, conns = feedforward
    nodes_before, conns_before = set(nodes), set(conns)

    host_graph.topological_sort(nodes, conns)

    assert nodes == nodes_before
    assert conns == conns_before

    # The list spelling must survive too, including a duplicated connection row:
    # normalising to a set may not be written back into the caller's list.
    node_list = list(nodes)
    conn_list = sorted(conns) + [(2, 3)]
    node_list_before, conn_list_before = list(node_list), list(conn_list)

    host_graph.topological_sort(node_list, conn_list)

    assert node_list == node_list_before
    assert conn_list == conn_list_before


def test_topological_sort_raises_graph_error_on_a_cycle(host_graph, cyclic, errors):
    """Invariant: a cycle raises ``GraphError`` once no zero-in-degree node remains.

    Oracle: the documented error mapping (``ValueError`` in the pre-rewrite
    library); :func:`cyclic` deliberately has an acyclic prefix ``0`` so the
    failure happens *after* progress, which is the case a naive "first round is
    empty" check misses.
    """
    nodes, conns = cyclic

    with pytest.raises(errors.GraphError):
        host_graph.topological_sort(nodes, conns)

    # Dropping only the back edge ``2→1`` makes the same node set sortable, so
    # the error comes from the cycle and not from the node/edge payload.
    topo_order, topo_layers = host_graph.topological_sort({0, 1, 2}, {(0, 1), (1, 2)})
    assert topo_layers == [[0], [1], [2]]
    assert topo_order == [0, 1, 2]


def test_topological_sort_raises_graph_error_for_an_unknown_endpoint(
    host_graph, feedforward, errors
):
    """Invariant: a connection endpoint outside ``nodes`` raises ``GraphError``.

    Oracle: the documented clause; the graph is otherwise acyclic, so the error
    cannot come from the cycle path — it must be the endpoint validation.
    """
    nodes, conns = feedforward

    with pytest.raises(errors.GraphError):
        host_graph.topological_sort(nodes, set(conns) | {(2, 9)})

    with pytest.raises(errors.GraphError):
        host_graph.topological_sort(nodes, set(conns) | {(9, 2)})

    # The uninterrupted payload sorts, so each failure above is the invented key.
    topo_order, _ = host_graph.topological_sort(nodes, conns)
    assert topo_order == [0, 1, 2, 5, 3, 4]


# --------------------------------------------------------------------------- #
# find_useful_nodes
# --------------------------------------------------------------------------- #


def test_find_useful_nodes_is_the_backward_closure_of_the_outputs(host_graph, feedforward):
    """Invariant: the backward closure of output ``4`` is ``{0, 1, 2, 3, 4}``.

    Oracle: hand enumeration of :func:`feedforward`; the fixpoint starts at the
    output, adds ``3``, then ``2``, then both inputs, and stops. ``5`` is *not*
    added: nothing useful depends on it.
    """
    nodes, conns = feedforward

    useful = host_graph.find_useful_nodes(nodes, conns, [4])

    assert isinstance(useful, set)
    assert useful == {0, 1, 2, 3, 4}
    assert 5 not in useful
    assert all(type(key) is int for key in useful)

    # Declaring the dangling node as a second output widens the closure to the
    # whole genome, which pins the seed handling (not just the edge walk).
    assert host_graph.find_useful_nodes(nodes, conns, [4, 5]) == {0, 1, 2, 3, 4, 5}

    # A genome whose only edge is the dangling one leaves the output isolated:
    # the closure is exactly the seeded output, not the node set.
    assert host_graph.find_useful_nodes(nodes, {(1, 5)}, [4]) == {4}


def test_find_useful_nodes_handles_a_cycle_through_the_output(host_graph, cyclic):
    """Invariant: a recurrent genome's closure still terminates and includes the cycle.

    Oracle: the documented "cycles need no special handling" — monotonicity bounds
    the fixpoint. :func:`cyclic` with output ``2`` must return ``{0, 1, 2}``, i.e.
    every node of the cycle, in finitely many rounds.
    """
    nodes, conns = cyclic

    assert host_graph.find_useful_nodes(nodes, conns, [2]) == {0, 1, 2}
    # Seeding inside the cycle reaches the same closure.
    assert host_graph.find_useful_nodes(nodes, conns, [1]) == {0, 1, 2}
    # No connection points into ``0``, so seeding the prefix selects only it.
    assert host_graph.find_useful_nodes(nodes, conns, [0]) == {0}


def test_find_useful_nodes_rejects_an_output_outside_the_genome(
    host_graph, feedforward, errors
):
    """Invariant: an ``output_idx`` key absent from ``nodes`` raises ``ConfigError``.

    Oracle: the documented clause — the caller's genome and its declared outputs
    disagree, which is a configuration error rather than a graph one.
    """
    nodes, conns = feedforward

    with pytest.raises(errors.ConfigError) as excinfo:
        host_graph.find_useful_nodes(nodes, conns, [9])
    assert "9" in str(excinfo.value)

    # One bad key among good ones is still a configuration error.
    with pytest.raises(errors.ConfigError):
        host_graph.find_useful_nodes(nodes, conns, [4, 9])


# --------------------------------------------------------------------------- #
# check_cycles
# --------------------------------------------------------------------------- #


def test_check_cycles_is_true_for_a_self_loop(host_graph, feedforward):
    """Invariant: ``from_idx == to_idx`` always reports a cycle.

    Oracle: hand enumeration — a self-loop closes a cycle by definition, and the
    legacy code answered ``True`` here as well.
    """
    nodes, conns = feedforward
    conns_before = set(conns)

    for key in (0, 3, 5):
        assert host_graph.check_cycles(nodes, conns, key, key) is True

    # The candidate edge is not inserted into ``conns``.
    assert conns == conns_before


def test_check_cycles_is_true_for_a_back_edge(host_graph, feedforward):
    """Invariant: ``4→0`` closes a cycle because ``0`` already reaches ``4``.

    Oracle: the transitive closure along ``0→2→3→4`` in :func:`feedforward`; the
    candidate edge is *not* inserted into ``conns``, so an implementation that
    mutates its argument would also fail
    :func:`test_topological_sort_does_not_mutate_its_arguments`.
    """
    nodes, conns = feedforward
    conns_before = set(conns)

    # 0 → 2 → 3 → 4 exists, so adding 4 → 0 would close the loop.
    assert host_graph.check_cycles(nodes, conns, 4, 0) is True
    # The shorter back edge 2 → 0 closes the 0 → 2 loop as well.
    assert host_graph.check_cycles(nodes, conns, 2, 0) is True
    # 5 is a leaf of 1, so 5 → 1 would close the two-node loop.
    assert host_graph.check_cycles(nodes, conns, 5, 1) is True

    assert conns == conns_before


def test_check_cycles_is_false_for_a_forward_edit(host_graph, feedforward):
    """Invariant: ``0→4`` is a new edge in an acyclic graph, so no cycle is closed.

    Oracle: hand enumeration — ``0`` reaches ``4`` already, but nothing reaches
    ``0``, so the added edge cannot return to its source.
    """
    nodes, conns = feedforward
    conns_before = set(conns)

    assert host_graph.check_cycles(nodes, conns, 0, 4) is False
    # Both endpoints out of the same input, and a fresh dangling edge, are fine.
    assert host_graph.check_cycles(nodes, conns, 1, 3) is False
    assert host_graph.check_cycles(nodes, conns, 4, 5) is False

    assert conns == conns_before


def test_check_cycles_is_false_for_an_existing_edge(host_graph, feedforward):
    """Invariant: re-adding ``2→3`` reports no cycle (it is a duplicate, not a loop).

    Oracle: the documented legacy behaviour — only a *new* back path closes a
    cycle; duplicate detection is the caller's job.
    """
    nodes, conns = feedforward

    for edge in sorted(conns):
        assert host_graph.check_cycles(nodes, conns, edge[0], edge[1]) is False


def test_check_cycles_rejects_an_unknown_endpoint(host_graph, feedforward, errors):
    """Invariant: ``from_idx``/``to_idx`` outside ``nodes`` raises ``ConfigError``.

    Oracle: the documented clause; a mutation onto an invented key is a caller
    bug that must not be answered with a boolean.
    """
    nodes, conns = feedforward

    with pytest.raises(errors.ConfigError) as excinfo:
        host_graph.check_cycles(nodes, conns, 9, 2)
    assert "9" in str(excinfo.value)

    with pytest.raises(errors.ConfigError) as excinfo:
        host_graph.check_cycles(nodes, conns, 2, 9)
    assert "9" in str(excinfo.value)

    # A self-loop on an invented key is an unknown endpoint, not a cycle.
    with pytest.raises(errors.ConfigError):
        host_graph.check_cycles(nodes, conns, 9, 9)


# --------------------------------------------------------------------------- #
# node_depth
# --------------------------------------------------------------------------- #


def test_node_depth_is_the_hand_enumerated_longest_path(host_graph, feedforward):
    """Invariant: ``{0: 0, 1: 0, 2: 1, 3: 2, 4: 3, 5: 1}`` with inputs at depth ``0``.

    Oracle: hand enumeration of :func:`feedforward`, where a *shortest*-path
    implementation would give the same answers — which is why the fixture contains
    a node with two paths of different length locally: ``2`` is reached from two
    inputs but its depth is still ``max``, and the output sits three edges below
    the inputs. Every key of ``nodes`` must appear exactly once.
    """
    nodes, conns = feedforward

    depth = host_graph.node_depth(nodes, conns, [0, 1])

    assert isinstance(depth, dict)
    assert depth == {0: 0, 1: 0, 2: 1, 3: 2, 4: 3, 5: 1}
    assert sorted(depth) == sorted(nodes)
    assert all(
        type(key) is int and type(value) is int for key, value in depth.items()
    )

    # Longest path, not shortest: adding the shortcut 0→4 would leave 4 at depth
    # 3 (longest 0→2→3→4) while the shortest path is 1. A shortest-path walk
    # would answer 1 here.
    shortcut_depth = host_graph.node_depth(nodes, set(conns) | {(0, 4)}, [0, 1])
    assert shortcut_depth[4] == 3
    assert shortcut_depth == {0: 0, 1: 0, 2: 1, 3: 2, 4: 3, 5: 1}


def test_node_depth_raises_graph_error_on_a_cycle_or_an_unreachable_node(
    host_graph, cyclic, errors
):
    """Invariant: a cycle or an input-unreachable node raises ``GraphError``.

    Oracle: the documented divergence from the schedule (which encodes both cases
    as ``I_INF``): the host walk cannot assign a depth, so it raises. Covered for
    a cyclic genome, for an unreachable root that feeds a further node, and for a
    whole disconnected component.
    """
    nodes, conns = cyclic

    with pytest.raises(errors.GraphError):
        host_graph.node_depth(nodes, conns, [0])

    # Acyclic, but ``1`` feeds ``2`` and no input reaches either of them: the
    # walk stops at the first root it cannot pin and names it.
    unreachable_nodes = {0, 1, 2}
    with pytest.raises(errors.GraphError) as excinfo:
        host_graph.node_depth(unreachable_nodes, {(1, 2)}, [0])
    assert "1" in str(excinfo.value)

    # Neither ``1`` nor ``2`` is fed; the call still raises rather than returning
    # a hole, and the node it names is one of the two unreachable ones.
    with pytest.raises(errors.GraphError) as excinfo:
        host_graph.node_depth({0, 1, 2}, set(), [0, 1])
    assert any(key in str(excinfo.value) for key in ("1", "2"))

    # A whole disconnected component is unreachable for the same reason.
    with pytest.raises(errors.GraphError) as excinfo:
        host_graph.node_depth({0, 1, 2, 3}, {(2, 3)}, [0, 1])
    assert "2" in str(excinfo.value)


def test_node_depth_rejects_an_empty_or_unknown_input_set(
    host_graph, feedforward, errors
):
    """Invariant: an empty ``input_idx`` or a key absent from ``nodes`` raises ``ConfigError``.

    Oracle: the documented clause; with no pinned level ``0`` the relaxation has
    no starting point, so an empty input set is a configuration error rather than
    an empty result.
    """
    nodes, conns = feedforward

    with pytest.raises(errors.ConfigError) as excinfo:
        host_graph.node_depth(nodes, conns, [])

    with pytest.raises(errors.ConfigError) as excinfo:
        host_graph.node_depth(nodes, conns, [9])
    assert "9" in str(excinfo.value)

    # One bad key among good ones is still a configuration error.
    with pytest.raises(errors.ConfigError):
        host_graph.node_depth(nodes, conns, [0, 9])
