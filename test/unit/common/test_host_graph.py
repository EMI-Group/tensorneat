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

Backend-independent (no ``evoxc`` op is called); no ``jax``/``torch``, no network.
"""

from __future__ import annotations

import importlib

import numpy as np
import pytest


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
    pytest.skip(
        "not implemented: decode_graph {key: attrs} / {(in, out)} vs the literal payload rows "
        "(owner: common/host_graph.decode_graph)"
    )


def test_decode_graph_collapses_duplicate_connection_pairs(host_graph):
    """Invariant: the same ``(in_key, out_key)`` twice yields one element of the returned set.

    Oracle: the documented "a set makes the topology"; the schedule, not this
    module, decides whether a duplicate is legal — so the decoded set must simply
    be smaller than the row count.
    """
    pytest.skip(
        "not implemented: duplicate connection pairs collapse into one set element "
        "(owner: common/host_graph.decode_graph)"
    )


def test_decode_graph_rejects_a_duplicate_node_key(host_graph, gene_rows):
    """Invariant: two used node rows with the same key raise ``GraphError``.

    Oracle: the documented ``Raises`` clause; a duplicate key makes the
    key→attribute map lossy, so it is a genome-structure error and not a warning.
    """
    pytest.skip(
        "not implemented: GraphError for a duplicated node key "
        "(owner: common/host_graph.decode_graph)"
    )


def test_decode_graph_rejects_a_connection_to_an_absent_node(host_graph, gene_rows):
    """Invariant: an endpoint that names no used node row raises ``GraphError``.

    Oracle: the documented clause; the payload is truncated so one endpoint key
    has no node row, which is what a truncated connection list looks like.
    """
    pytest.skip(
        "not implemented: GraphError for a connection endpoint without a node row "
        "(owner: common/host_graph.decode_graph)"
    )


def test_decode_graph_rejects_a_bad_fixed_prefix(host_graph, gene_rows):
    """Invariant: ``node_attrs < 1`` / ``conn_attrs < 2`` and a too-short row raise ``ConfigError``.

    Oracle: the documented widths (``fixed_attrs == ["idx"]`` for nodes,
    ``["input_index", "output_index"]`` for connections); a row shorter than its
    fixed prefix cannot even be keyed, so it is a configuration error surfaced at
    export time.
    """
    pytest.skip(
        "not implemented: ConfigError for a bad node_attrs/conn_attrs width or a short row "
        "(owner: common/host_graph.decode_graph)"
    )


# --------------------------------------------------------------------------- #
# topological_sort
# --------------------------------------------------------------------------- #


def test_topological_sort_returns_layers_and_their_concatenation(host_graph, feedforward):
    """Invariant: ``topo_layers`` is the literal ``[[0, 1], [2, 5], [3], [4]]``.

    Oracle: hand enumeration of :func:`feedforward`; each layer is sorted
    ascending, layer ``0`` is the in-degree-0 set (the inputs), ``topo_order`` is
    their concatenation and every node appears exactly once across the partition.
    """
    pytest.skip(
        "not implemented: topological_sort layers/order vs the literal [[0, 1], [2, 5], [3], [4]] "
        "(owner: common/host_graph.topological_sort)"
    )


def test_topological_sort_is_deterministic_regardless_of_input_order(host_graph, feedforward):
    """Invariant: the same topology given as lists in different orders yields the same layers.

    Oracle: the ascending-within-layer rule of the docstring, checked by feeding
    the reversed node list and the reversed connection list as well — set
    iteration order must not leak into the result, because visualisation and
    formula output are diffed between runs.
    """
    pytest.skip(
        "not implemented: topological_sort identical for list/set/reversed inputs "
        "(owner: common/host_graph.topological_sort)"
    )


def test_topological_sort_does_not_mutate_its_arguments(host_graph, feedforward):
    """Invariant: ``nodes`` and ``conns`` are unchanged after the call.

    Oracle: a copy taken before, compared after; the pre-rewrite version copied
    its arguments, and a caller that reuses its genome between rounds would
    otherwise lose edges.
    """
    pytest.skip(
        "not implemented: topological_sort leaves nodes/conns untouched "
        "(owner: common/host_graph.topological_sort)"
    )


def test_topological_sort_raises_graph_error_on_a_cycle(host_graph, cyclic):
    """Invariant: a cycle raises ``GraphError`` once no zero-in-degree node remains.

    Oracle: the documented error mapping (``ValueError`` in the pre-rewrite
    library); :func:`cyclic` deliberately has an acyclic prefix ``0`` so the
    failure happens *after* progress, which is the case a naive "first round is
    empty" check misses.
    """
    pytest.skip(
        "not implemented: GraphError for a cyclic genome "
        "(owner: common/host_graph.topological_sort)"
    )


def test_topological_sort_raises_graph_error_for_an_unknown_endpoint(host_graph, feedforward):
    """Invariant: a connection endpoint outside ``nodes`` raises ``GraphError``.

    Oracle: the documented clause; the graph is otherwise acyclic, so the error
    cannot come from the cycle path — it must be the endpoint validation.
    """
    pytest.skip(
        "not implemented: GraphError for an endpoint outside nodes "
        "(owner: common/host_graph.topological_sort)"
    )


# --------------------------------------------------------------------------- #
# find_useful_nodes
# --------------------------------------------------------------------------- #


def test_find_useful_nodes_is_the_backward_closure_of_the_outputs(host_graph, feedforward):
    """Invariant: the backward closure of output ``4`` is ``{0, 1, 2, 3, 4}``.

    Oracle: hand enumeration of :func:`feedforward`; the fixpoint starts at the
    output, adds ``3``, then ``2``, then both inputs, and stops. ``5`` is *not*
    added: nothing useful depends on it.
    """
    pytest.skip(
        "not implemented: find_useful_nodes closure {0, 1, 2, 3, 4} for output 4 "
        "(owner: common/host_graph.find_useful_nodes)"
    )


def test_find_useful_nodes_handles_a_cycle_through_the_output(host_graph, cyclic):
    """Invariant: a recurrent genome's closure still terminates and includes the cycle.

    Oracle: the documented "cycles need no special handling" — monotonicity bounds
    the fixpoint. :func:`cyclic` with output ``2`` must return ``{0, 1, 2}``, i.e.
    every node of the cycle, in finitely many rounds.
    """
    pytest.skip(
        "not implemented: find_useful_nodes terminates on a cycle and includes all of it "
        "(owner: common/host_graph.find_useful_nodes)"
    )


def test_find_useful_nodes_rejects_an_output_outside_the_genome(host_graph, feedforward):
    """Invariant: an ``output_idx`` key absent from ``nodes`` raises ``ConfigError``.

    Oracle: the documented clause — the caller's genome and its declared outputs
    disagree, which is a configuration error rather than a graph one.
    """
    pytest.skip(
        "not implemented: ConfigError for an output key outside nodes "
        "(owner: common/host_graph.find_useful_nodes)"
    )


# --------------------------------------------------------------------------- #
# check_cycles
# --------------------------------------------------------------------------- #


def test_check_cycles_is_true_for_a_self_loop(host_graph, feedforward):
    """Invariant: ``from_idx == to_idx`` always reports a cycle.

    Oracle: hand enumeration — a self-loop closes a cycle by definition, and the
    legacy code answered ``True`` here as well.
    """
    pytest.skip(
        "not implemented: check_cycles True for a self-loop "
        "(owner: common/host_graph.check_cycles)"
    )


def test_check_cycles_is_true_for_a_back_edge(host_graph, feedforward):
    """Invariant: ``4→0`` closes a cycle because ``0`` already reaches ``4``.

    Oracle: the transitive closure along ``0→2→3→4`` in :func:`feedforward`; the
    candidate edge is *not* inserted into ``conns``, so an implementation that
    mutates its argument would also fail
    :func:`test_topological_sort_does_not_mutate_its_arguments`.
    """
    pytest.skip(
        "not implemented: check_cycles True for the back edge 4 -> 0 "
        "(owner: common/host_graph.check_cycles)"
    )


def test_check_cycles_is_false_for_a_forward_edit(host_graph, feedforward):
    """Invariant: ``0→4`` is a new edge in an acyclic graph, so no cycle is closed.

    Oracle: hand enumeration — ``0`` reaches ``4`` already, but nothing reaches
    ``0``, so the added edge cannot return to its source.
    """
    pytest.skip(
        "not implemented: check_cycles False for the forward edge 0 -> 4 "
        "(owner: common/host_graph.check_cycles)"
    )


def test_check_cycles_is_false_for_an_existing_edge(host_graph, feedforward):
    """Invariant: re-adding ``2→3`` reports no cycle (it is a duplicate, not a loop).

    Oracle: the documented legacy behaviour — only a *new* back path closes a
    cycle; duplicate detection is the caller's job.
    """
    pytest.skip(
        "not implemented: check_cycles False for an already existing edge "
        "(owner: common/host_graph.check_cycles)"
    )


def test_check_cycles_rejects_an_unknown_endpoint(host_graph, feedforward):
    """Invariant: ``from_idx``/``to_idx`` outside ``nodes`` raises ``ConfigError``.

    Oracle: the documented clause; a mutation onto an invented key is a caller
    bug that must not be answered with a boolean.
    """
    pytest.skip(
        "not implemented: ConfigError for an unknown candidate endpoint "
        "(owner: common/host_graph.check_cycles)"
    )


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
    pytest.skip(
        "not implemented: node_depth dict equals the literal longest-path depths "
        "(owner: common/host_graph.node_depth)"
    )


def test_node_depth_raises_graph_error_on_a_cycle_or_an_unreachable_node(
    host_graph, cyclic
):
    """Invariant: a cycle or an input-unreachable node raises ``GraphError``.

    Oracle: the documented divergence from the schedule (which encodes both cases
    as ``I_INF``): the host walk cannot assign a depth, so it raises. Covered for
    a cyclic genome and for a node fed only by a hidden node with no path from the
    inputs.
    """
    pytest.skip(
        "not implemented: GraphError for a cycle and for an unreachable node in node_depth "
        "(owner: common/host_graph.node_depth)"
    )


def test_node_depth_rejects_an_empty_or_unknown_input_set(host_graph, feedforward):
    """Invariant: an empty ``input_idx`` or a key absent from ``nodes`` raises ``ConfigError``.

    Oracle: the documented clause; with no pinned level ``0`` the relaxation has
    no starting point, so an empty input set is a configuration error rather than
    an empty result.
    """
    pytest.skip(
        "not implemented: ConfigError for empty/unknown input_idx in node_depth "
        "(owner: common/host_graph.node_depth)"
    )
