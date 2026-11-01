"""Unit tests for ``tensorneat.common.tensorization`` — the execution schedule.

What this module proves
-----------------------
``derive_schedule`` turns a ragged population of evolved topologies into the
static arrays every forward pass is built on: node levels (a level-synchronous
topological order), the ``(P, N, K)`` incoming-edge table, the ``(P, C)``
``src``/``dst``/``edge_valid`` triple, and the input/output slot vectors. The
tests check *semantics*, not implementation strategy: the expected level of a
node is its longest-path depth from the inputs, the expected window of a node is
the set of its incoming connection rows in row order, and both are enumerated by
hand or by an oracle written independently of the kernel.

Oracles used here (never the kernel itself)
-------------------------------------------
* ``support.oracle.schedule`` — the reference derivation from
  ``../../support/CONTEXT.md`` (dense key→slot table, in-group rank, counting
  trick, longest-path levels). It is compared field by field.
* ``tensorneat.common.host_graph.node_depth`` — the *host* longest-path walk,
  which the module docstring names as the agreement hook for this suite. The
  schedule and the host walk must report the same depth for a feedforward
  genome, and they must diverge on a recurrent one (``level == I_INF`` in the
  schedule where the host walk raises ``GraphError``).
* hand enumeration — the two-node-in/one-node-out chain built row by row in
  :func:`tiny`, whose levels, windows and slots are written out literally.

Gotchas this module encodes
---------------------------
* Keys are ``integer-valued`` and *exact*; an absent node row is a sentinel, and
  the two sentinels in the package are typed (``I_INF`` in integer arrays,
  ``NaN`` in float arrays). ``derive_schedule`` documents its key columns as
  ``float32`` with ``NaN`` padding while the gene payload
  (``genome/utils.py``, ``support/payloads.py``) stores ``int32`` keys with
  ``I_INF`` padding, so :func:`_float_keys` reconciles the two in exactly one
  place — see the report to the parent, this is a live doc conflict.
* ``derive_schedule``'s statics are **keyword-only**, and ``evoxc.build`` passes
  only positional arguments, so the kernel is bound with
  ``functools.partial(...fn, **statics)`` (:func:`_schedule`). A ``Defn`` object
  is not callable, hence the ``.fn`` unwrap.
* ``edge_overflow`` is *data*, never an exception: a too-small ``max_in_degree``
  is reported per individual and the retained columns are the lowest in-group
  ranks. Nothing in this module expects a raise for that case.

Backend is ``"numpy"`` throughout (the shared runner's default); no optional
dependency, no ``jax``/``torch`` import at module scope, no network.
"""

from __future__ import annotations

from functools import partial

import numpy as np
import pytest

from support.runner import run_defn

# --------------------------------------------------------------------------- #
# the schedule under test, reached through the `common` fixture
# --------------------------------------------------------------------------- #


def _float_keys(keys, i_inf):
    """Reconcile the two key sentinels: ``int32``/``I_INF`` payload → ``float32``/``NaN`` input.

    ``derive_schedule`` documents ``node_keys``/``conn_in``/``conn_out`` as
    ``float32`` with ``NaN`` marking an absent row, while the gene payload keeps
    ``int32`` keys and marks an absent row with ``I_INF`` (an ``int32`` sentinel
    that cannot survive a cast to ``float32``: it becomes ``2.147e9``). This is
    the single place where the two conventions meet, so a change to either one is
    a one-line change here and a visible failure in every test below.
    """
    keys = np.asarray(keys, dtype=np.int32)
    converted = keys.astype(np.float32)
    converted[keys == i_inf] = np.nan
    return converted


def _io_keys(payload, num_inputs, num_outputs, i_inf):
    """Input/output node **keys**, read off the payload's documented row invariant.

    ``initialize`` writes nodes in key order and never moves the input rows
    (``0..num_inputs-1``) or the output rows (``max_nodes-num_outputs..``), so the
    first row of the batch identifies every population member's input and output
    keys (``genome/utils.py``, ``GenomeTensors``).
    """
    node_keys = np.asarray(payload.node_keys)
    header = node_keys[0]
    max_nodes = node_keys.shape[1]
    return (
        _float_keys(header[:num_inputs], i_inf),
        _float_keys(header[max_nodes - num_outputs :], i_inf),
    )


def _schedule(common, payload, num_inputs=2, num_outputs=1, *, max_in_degree=None, key_span=None):
    """Derive one payload's schedule through the shared runner and return it as numpy.

    Args:
        common: The ``tensorneat.common`` package fixture.
        payload: One ``support.payloads.Payload`` (a whole population).
        num_inputs, num_outputs: The genome configuration the payload was built
            for; only their counts matter, and the rows they name are stable.
        max_in_degree: ``K``; defaults to ``max_nodes`` (the largest in-degree a
            population can express, so nothing overflows unless a test asks).
        key_span: ``S``; defaults to ``max(node key) + 1`` on the host, the
            smallest dense key→slot table that can hold the payload.

    Notes:
        The five statics of ``derive_schedule`` are keyword-only, and
        ``evoxc.build`` mirrors a *positional* call, so they are bound with
        ``functools.partial`` — over ``derive_schedule.fn``, because a ``Defn``
        is deliberately not callable.
    """
    i_inf = common.I_INF
    node_keys = _float_keys(payload.node_keys, i_inf)
    conn_in = _float_keys(np.asarray(payload.conn_keys)[..., 0], i_inf)
    conn_out = _float_keys(np.asarray(payload.conn_keys)[..., 1], i_inf)
    input_keys, output_keys = _io_keys(payload, num_inputs, num_outputs, i_inf)

    max_nodes = node_keys.shape[1]
    if max_in_degree is None:
        max_in_degree = max_nodes
    if key_span is None:
        used = np.asarray(payload.node_keys)[np.asarray(payload.node_keys) != i_inf]
        key_span = int(used.max()) + 1

    statics = dict(
        pop_size=node_keys.shape[0],
        max_nodes=max_nodes,
        max_conns=conn_in.shape[1],
        max_in_degree=int(max_in_degree),
        key_span=int(key_span),
    )
    kernel = partial(common.tensorization.derive_schedule.fn, **statics)
    return run_defn(kernel, node_keys, conn_in, conn_out, input_keys, output_keys)


def _host_topology(payload, i_inf, member=0):
    """One population member's topology as plain Python (``set[int]``, ``set[tuple[int, int]]``).

    This is what ``host_graph`` consumes. The payload already stores integer keys,
    so no float decoding is needed here; ``host_graph.decode_graph`` documents a
    ``float32``/``NaN`` payload instead (a doc conflict reported upward, and the
    reason this helper exists rather than calling it).
    """
    node_keys = np.asarray(payload.node_keys)[member]
    conn_keys = np.asarray(payload.conn_keys)[member]
    nodes = {int(key) for key in node_keys if key != i_inf}
    conns = {
        (int(pair[0]), int(pair[1])) for pair in conn_keys if pair[0] != i_inf
    }
    return nodes, conns


# --------------------------------------------------------------------------- #
# fixtures: the canonical hand-built genomes of ../../support/payloads.py
# --------------------------------------------------------------------------- #


@pytest.fixture
def tiny(payloads):
    """A two-in/one-hidden/one-out chain built row by row (keys ``0,1,2,3``).

    Topology, literally: ``0 → 2``, ``1 → 2``, ``2 → 3``; keys ``0`` and ``1`` are
    the inputs, key ``3`` is the output. Its schedule is therefore fully
    enumerable by hand — levels ``[0, 0, 1, 2]``, node ``2``'s in-edge window
    ``((row 0, src 0), (row 1, src 1))``, node ``3``'s ``((row 2, src 2),)``, and
    ``I_INF`` everywhere else.
    """
    payload = payloads.new_payload(1, 4, 3)
    for row, key in enumerate((0, 1, 2, 3)):
        payloads.set_node(payload, 0, row, key)
    payloads.set_conn(payload, 0, 0, 0, 2)
    payloads.set_conn(payload, 0, 1, 1, 2)
    payloads.set_conn(payload, 0, 2, 2, 3)
    return payload


@pytest.fixture
def chain_payload(payloads):
    """The canonical dense chain (a single path from an input to the output)."""
    return payloads.chain_payload()


@pytest.fixture
def skip_payload(payloads):
    """The canonical topology with a node bypassed by its consumers."""
    return payloads.skip_payload()


@pytest.fixture
def isolated_payload(payloads):
    """The canonical topology with a node that has no edge at all."""
    return payloads.isolated_payload()


@pytest.fixture
def ring_payload(payloads):
    """The canonical recurrent ring (deliberately cyclic)."""
    return payloads.ring_payload()


# --------------------------------------------------------------------------- #
# shape contract
# --------------------------------------------------------------------------- #


def test_transformed_fields_have_the_documented_shapes_and_dtypes(
    common, chain_payload
):
    """Invariant: every ``Transformed`` field is batch-first with the documented shape and dtype.

    Oracle: the shapes spelled out from the payload alone — ``src``/``dst``/
    ``edge_valid`` are ``(P, C)``, ``in_conn``/``in_src`` ``(P, N, K)``,
    ``level``/``topo_key`` ``(P, N)``, ``input_slot``/``output_slot`` ``(P, I)``/
    ``(P, O)``, ``edge_overflow`` ``(P,)`` — with ``int32`` for the integer
    fields and ``bool`` for ``edge_valid``. ``Transformed`` holds arrays only, so
    a Python static leaking into it would be a violation too.
    """
    pytest.skip(
        "not implemented: Transformed field shapes/dtypes from payload shapes + "
        "num_inputs/num_outputs "
        "(owner: common/tensorization.derive_schedule)"
    )


# --------------------------------------------------------------------------- #
# hand-enumerated schedule of the tiny chain
# --------------------------------------------------------------------------- #


def test_tiny_chain_levels_are_hand_enumerated(common, tiny):
    """Invariant: the tiny chain's levels are literally ``[0, 0, 1, 2]``.

    Oracle: hand enumeration of the documented topology (``0→2``, ``1→2``,
    ``2→3``): the inputs are level ``0``, the hidden node is one edge below them,
    the output is two. No oracle module is involved — this is the "small hand-
    built genome" case.
    """
    pytest.skip(
        "not implemented: tiny chain level == [[0, 0, 1, 2]] "
        "(owner: common/tensorization.node_levels)"
    )


def test_tiny_chain_incoming_windows_are_hand_enumerated(common, tiny):
    """Invariant: the windows are literally ``{2: [(0, 0), (1, 1)], 3: [(2, 2)]}`` (row, src).

    Oracle: hand enumeration of the same three-connection topology. Column order
    is connection-row order (the documented "dense columns ``0..d-1``" plus the
    in-group rank rule), and every other column is ``I_INF`` in **both**
    ``in_conn`` and ``in_src``.
    """
    pytest.skip(
        "not implemented: tiny chain in_conn/in_src windows in connection-row order "
        "(owner: common/tensorization.incoming_table)"
    )


def test_tiny_chain_topology_key_and_slots_are_hand_enumerated(common, tiny):
    """Invariant: ``topo_key == level*N + slot``, ``input_slot == [0, 1]``, ``output_slot == [3]``.

    Oracle: hand enumeration; ``topo_key`` is documented as a cheap total order,
    so the equality is exact, and the slot vectors follow ``input_keys``/
    ``output_keys`` order.
    """
    pytest.skip(
        "not implemented: tiny chain topo_key/input_slot/output_slot literals "
        "(owner: common/tensorization.derive_schedule)"
    )


# --------------------------------------------------------------------------- #
# levels
# --------------------------------------------------------------------------- #


def test_input_nodes_are_level_zero(common, chain_payload):
    """Invariant: every row named by ``input_keys`` has ``level == 0``.

    Oracle: the documented pinning of the input nodes to level ``0`` (they are the
    relaxation's initial values, not a computed depth), checked on the rows
    ``0..num_inputs-1`` that the payload's row invariant fixes.
    """
    pytest.skip(
        "not implemented: level[p, :num_inputs] == 0 on the canonical chain "
        "(owner: common/tensorization.node_levels)"
    )


def test_levels_obey_the_longest_path_recursion(common, chain_payload):
    """Invariant: ``level[dst] == 1 + max(level[src] for every edge into dst)``.

    Oracle: the recursion the docstring states, evaluated **from the decoded edge
    list** (``_host_topology``) rather than from any schedule output; it also
    implies the weaker (but separately valuable) "every edge goes strictly
    upwards", which is what makes the level sweep a topological order.
    """
    pytest.skip(
        "not implemented: level recursion over the payload's real edges "
        "(owner: common/tensorization.node_levels)"
    )


def test_absent_padding_rows_get_level_i_inf(common, chain_payload):
    """Invariant: a node row that no node occupies (padding) has ``level == I_INF``.

    Oracle: the documented "``level`` is ``I_INF`` iff node ``i`` is absent"; the
    canonical payload pads ``max_nodes`` beyond its real nodes, so the padding
    rows are known from the empty key column.
    """
    pytest.skip(
        "not implemented: absent rows carry level I_INF (and topo_key I_INF) "
        "(owner: common/tensorization.node_levels)"
    )


def test_isolated_node_gets_level_i_inf(common, isolated_payload):
    """Invariant: a node with no incoming edge is unreachable, hence ``level == I_INF``.

    Oracle: the documented "unreachable from the inputs" case. An isolated node
    is *present* (its key row is real) but can never be relaxed, so its level is
    the sentinel — not ``0``, which would put it in the first sweep and make it
    look like an input.
    """
    pytest.skip(
        "not implemented: isolated node level == I_INF, present but unreachable "
        "(owner: common/tensorization.node_levels)"
    )


def test_cyclic_nodes_get_level_i_inf_and_the_host_walk_raises(common, ring_payload):
    """Invariant: cyclic nodes carry ``I_INF`` in ``level``, where ``host_graph.node_depth`` raises.

    Oracle: two independent statements of the same fact — the in-graph schedule
    reports an unrelaxable node as data (``I_INF``, the documented divergence from
    a raise), while the host walk raises ``GraphError`` for a genome whose
    execution is no longer feedforward. A recurrent genome therefore has *no*
    depth agreement, which the two assertions pin down.
    """
    pytest.skip(
        "not implemented: ring levels == I_INF while host_graph.node_depth raises GraphError "
        "(owner: common/tensorization.node_levels, common/host_graph.node_depth)"
    )


def test_level_agrees_with_host_node_depth_on_a_feedforward_genome(
    common, chain_payload, skip_payload
):
    """Invariant: for a feedforward genome ``{key: level}`` equals ``host_graph.node_depth``.

    Oracle: ``host_graph.node_depth(nodes, conns, input_idx)`` — the host
    longest-path walk over the same decoded topology, which the module docstring
    of ``node_depth`` names as the agreement hook for this suite. Compared on two
    shapes with different depths (a chain and a bypass), so an off-by-one or a
    "shortest path" implementation cannot pass.
    """
    pytest.skip(
        "not implemented: level map == host_graph.node_depth for chain and bypass payloads "
        "(owner: common/tensorization.node_levels vs common/host_graph.node_depth)"
    )


# --------------------------------------------------------------------------- #
# incoming-edge windows
# --------------------------------------------------------------------------- #


def test_incoming_windows_are_dense_and_in_connection_row_order(common, chain_payload):
    """Invariant: node ``i``'s columns ``0..d-1`` hold its ``d`` incoming rows, ascending.

    Oracle: the connection list decoded from the payload (``_host_topology``)
    grouped by destination and sorted by row — the documented "dense prefix +
    in-group rank" rule. Catches a transposed or reversed window, and a window
    that is dense but ordered by source instead of by connection row.
    """
    pytest.skip(
        "not implemented: in_conn windows == the node's incoming rows in ascending order "
        "(owner: common/tensorization.incoming_table, edge_group_rank)"
    )


def test_incoming_windows_carry_the_source_slot_of_the_same_edge(common, chain_payload):
    """Invariant: ``in_src[p, i, k]`` is the source slot of the edge named by ``in_conn[p, i, k]``.

    Oracle: ``src[p, in_conn[p, i, k]]`` — the cross-reference the forward pass
    relies on when it gathers source *values* with ``in_src`` while it gathers
    connection *attributes* with ``in_conn``. A mismatch would feed a node an edge
    attribute of one connection with the value of another.
    """
    pytest.skip(
        "not implemented: in_src[i, k] == src[in_conn[i, k]] for every dense column "
        "(owner: common/tensorization.incoming_table)"
    )


def test_padded_columns_never_leak_into_values(common, chain_payload, skip_payload):
    """Invariant: ``in_conn`` is ``I_INF`` exactly where ``in_src`` is, padding only above ``d``.

    Oracle: the two invariances of the ``Transformed`` docstring, checked
    structurally: identical sentinel masks in the two tables, and, per node, "the
    dense prefix is ``d`` long" where ``d`` is counted from the decoded edges.
    This is the property that lets a forward pass gather with ``I_INF`` padding —
    a padded column that carries a real value (or an ``I_INF`` that hides a real
    edge) makes the gather read somebody else's data.
    """
    pytest.skip(
        "not implemented: in_conn/in_src sentinel masks agree "
        "and padding sits only above the real in-degree "
        "(owner: common/tensorization.incoming_table)"
    )


# --------------------------------------------------------------------------- #
# edge_overflow
# --------------------------------------------------------------------------- #


def test_edge_overflow_is_zero_when_the_bound_is_large_enough(common, chain_payload):
    """Invariant: ``edge_overflow == 0`` whenever ``max_in_degree`` covers every node's in-degree.

    Oracle: the in-degrees counted from the payload's edge list, compared with
    the statics ``K``; nothing may be reported as dropped when nothing was.
    """
    pytest.skip(
        "not implemented: edge_overflow == 0 for K == max_nodes "
        "(owner: common/tensorization.incoming_table)"
    )


def test_edge_overflow_counts_exactly_the_dropped_edges(common, chain_payload):
    """Invariant: with ``K`` one short of a node's in-degree ``d``, ``edge_overflow == d - K``.

    Oracle: the node's real in-degree ``d`` counted from the decoded edges — the
    overflow is *exactly* the number of edges beyond the bound, and the retained
    window is the first ``K`` rows in connection-row order (not an arbitrary
    subset). ``edge_overflow`` is data, so no exception is expected: this is the
    documented "the host decides to rebuild" path, and the retained window is what
    makes a lossy schedule still *deterministic*.
    """
    pytest.skip(
        "not implemented: edge_overflow == real_in_degree - K with the lowest ranks retained "
        "(owner: common/tensorization.incoming_table)"
    )


# --------------------------------------------------------------------------- #
# the whole schedule vs the independent numpy oracle
# --------------------------------------------------------------------------- #


def test_derive_schedule_matches_the_oracle_field_by_field(
    common, chain_payload, skip_payload, oracle
):
    """Invariant: every ``Transformed`` field equals ``support.oracle.schedule``'s.

    Oracle: the reference derivation of ``../../support/CONTEXT.md`` (dense
    key→slot table, in-group rank, counting trick, longest-path levels) on the
    same arrays. All ten fields are compared, so a correct ``level`` cannot hide a
    wrong ``topo_key`` — and the comparison runs on two topologies (chain,
    bypass) so a shape-specific coincidence is unlikely.
    """
    pytest.skip(
        "not implemented: all Transformed fields == support.oracle.schedule field by field "
        "(owner: common/tensorization.derive_schedule)"
    )


def test_topological_key_is_a_valid_execution_order(common, chain_payload):
    """Invariant: every valid edge goes strictly upwards, i.e. ``level[src] < level[dst]``.

    Oracle: the documented "executing levels ``0, 1, ...`` in order is a valid
    topological order". Checked over the payload's real edges only (absent and
    cyclic nodes carry ``I_INF`` and are excluded), plus the ``topo_key`` equal to
    ``level * N + slot``, which is what a sweep actually sorts on.
    """
    pytest.skip(
        "not implemented: level[src] < level[dst] for every valid edge "
        "and topo_key == level*N + slot "
        "(owner: common/tensorization.derive_schedule)"
    )


# --------------------------------------------------------------------------- #
# validate_schedule: the host-side invariant checker
# --------------------------------------------------------------------------- #


def test_validate_schedule_accepts_the_kernel_output(common, chain_payload, skip_payload):
    """Invariant: ``validate_schedule`` returns ``None`` for every schedule the kernel produces.

    Oracle: the kernel itself (the acceptance half) — plus the documented
    ``None`` return: a checker that returns a repaired schedule would silently
    hide the violation it was asked to report.
    """
    pytest.skip(
        "not implemented: validate_schedule returns None on chain and bypass schedules "
        "(owner: common/tensorization.validate_schedule)"
    )


def test_validate_schedule_rejects_in_conn_in_src_disagreement(common, chain_payload):
    """Invariant: a padded position in one table and a real index in the other is a ``ConfigError``.

    Oracle: the tampered schedule built with ``Transformed._replace`` — set one
    ``in_src`` padding cell to a real slot while ``in_conn`` keeps ``I_INF``. The
    two tables must agree on every sentinel, so this must raise, and the message
    must name the violated invariant (first violation wins).
    """
    pytest.skip(
        "not implemented: ConfigError for in_conn/in_src sentinel disagreement "
        "(owner: common/tensorization.validate_schedule)"
    )


def test_validate_schedule_rejects_i_inf_level_with_a_finite_topo_key(common, chain_payload):
    """Invariant: ``level == I_INF`` implies ``topo_key == I_INF``; a finite key raises.

    Oracle: the documented pair ``topo_key = level * N + slot`` plus its sentinel:
    an unavailable node cannot be given a place in the sweep. Tampered by
    replacing one padding row's ``topo_key`` with ``0``.
    """
    pytest.skip(
        "not implemented: ConfigError for level I_INF with a finite topo_key "
        "(owner: common/tensorization.validate_schedule)"
    )


def test_validate_schedule_rejects_negative_edge_overflow(common, chain_payload):
    """Invariant: a negative ``edge_overflow`` is a ``ConfigError``.

    Oracle: the field's meaning (a *count* of dropped edges); a negative count is
    impossible for any derivation and therefore always means a corrupted or
    hand-built schedule, which must not reach a rebuild decision.
    """
    pytest.skip(
        "not implemented: ConfigError for edge_overflow < 0 "
        "(owner: common/tensorization.validate_schedule)"
    )


def test_validate_schedule_rejects_an_out_of_range_input_slot(common, chain_payload):
    """Invariant: an ``input_slot`` outside ``[0, N)`` — and not ``I_INF`` — is a ``ConfigError``.

    Oracle: the documented "``input_slot``/``output_slot`` point inside ``[0, N)``
    or are the sentinel"; ``N`` itself is out of range (it is the first illegal
    index), which is the mistake an off-by-one derivation makes.
    """
    pytest.skip(
        "not implemented: ConfigError for input_slot == N (out of [0, N) and not I_INF) "
        "(owner: common/tensorization.validate_schedule)"
    )


# --------------------------------------------------------------------------- #
# the two leaf helpers of the derivation
# --------------------------------------------------------------------------- #


def test_key_slot_table_marks_absent_and_unknown_keys(common, chain_payload):
    """Invariant: a real key maps to its row slot; ``NaN`` and unknown keys map to ``I_INF``.

    Oracle: the payload's own key column — the expected table is
    ``{key: row}`` read straight off ``node_keys``. Also asserts the trailing
    dump column exists (width ``key_span + 1``), because an absent row must have
    somewhere to go that is not slot ``0``.
    """
    pytest.skip(
        "not implemented: key_slot_table {key: slot} vs the payload rows, width key_span + 1 "
        "(owner: common/tensorization.key_slot_table)"
    )


def test_slot_lookup_maps_nan_and_unknown_keys_to_i_inf(common, chain_payload):
    """Invariant: ``slot_lookup`` returns ``I_INF`` for ``NaN`` and for a key no node holds.

    Oracle: the table built by ``key_slot_table`` plus a hand-built query vector
    of ``[real key, NaN, key_span (out of range), unknown key]``; the sentinel must
    be produced even though the *index* had to be sanitized before the gather
    (clamping to ``0`` is a real node, so a leak shows up as slot ``0``).
    """
    pytest.skip(
        "not implemented: slot_lookup NaN/unknown -> I_INF and no index leak from sanitizing "
        "(owner: common/tensorization.slot_lookup)"
    )


def test_missing_output_key_raises_graph_error(common, chain_payload):
    """Invariant: an output key that no node holds is a ``GraphError``.

    Oracle: the documented ``Raises`` clause — a genome whose declared outputs
    are absent cannot be executed, and the error must be the package's
    ``GraphError`` (a ``TensorNEATError``), not an ``IndexError`` from a gather or
    a silent ``output_slot == I_INF``.
    """
    pytest.skip(
        "not implemented: GraphError when a required output key is missing from node_keys "
        "(owner: common/tensorization.derive_schedule)"
    )
