"""Unit tests for ``tensorneat.common.tensorization`` — the execution schedule.

``derive_schedule`` turns a ragged population of evolved topologies into the
static arrays every forward pass is built on: node levels (a level-synchronous
topological order), the ``(P, N, K)`` incoming-edge table, the ``(P, C)``
``src``/``dst``/``edge_valid`` triple and the input/output slot vectors. The tests
check *semantics*: a node's level is its longest-path depth from the inputs, a
node's window is its incoming connection rows in row order.

Oracles (never the kernel itself): ``support.oracle.schedule`` — the reference
derivation, returning the same twelve-field ``Transformed``, so all twelve fields
are compared; ``tensorneat.common.host_graph.node_depth`` — the host longest-path
walk, which must agree on a feedforward genome and raise where the schedule
reports ``I_INF``; hand enumeration for :func:`tiny`; and the payload's own arrays
(:func:`_decode`) for every window/overflow expectation.

Gotchas: keys are exact and come in two spellings — canonical ``int32``/``I_INF``
and alternate ``float32``/``NaN``, which must give the identical schedule
(:func:`_float_keys`); the five statics are positional-or-keyword, so
:func:`_schedule` passes them positionally (no ``partial``, no ``.fn``);
``Transformed`` has exactly twelve fields with ``level``/``overflow`` as read-only
properties; ``edge_overflow`` is *data*, never an exception; and ``level ==
I_INF`` is the schedule's way of reporting an absent, unreachable or cyclic node
where ``host_graph`` raises ``GraphError`` — a recurrent genome has no depth
agreement, by design.

Runtime: payloads and schedules are module-scoped fixtures (one ``evoxc.build``
per distinct topology/``K``/key spelling); derived arrays are read-only, so a
tampering test copies the field and rebuilds with ``Transformed._replace``.
Backend is ``"numpy"``, no ``jax``/``torch`` import, no network.
"""

from __future__ import annotations

import numpy as np
import pytest

from support.runner import run_defn

#: ``num_inputs`` / ``num_outputs`` of every payload fixture here (payloads defaults).
NUM_INPUTS = 2
NUM_OUTPUTS = 1

#: The twelve ``Transformed`` fields, in the order its docstring documents.
DOCUMENTED_FIELDS = (
    "src",
    "dst",
    "edge_valid",
    "in_conn",
    "in_src",
    "in_valid",
    "node_level",
    "node_exists",
    "topo_key",
    "input_slot",
    "output_slot",
    "edge_overflow",
)

#: The documented dtype of every field (integer slots vs. live-edge masks).
DOCUMENTED_DTYPES = {
    "src": np.int32,
    "dst": np.int32,
    "edge_valid": np.bool_,
    "in_conn": np.int32,
    "in_src": np.int32,
    "in_valid": np.bool_,
    "node_level": np.int32,
    "node_exists": np.bool_,
    "topo_key": np.int32,
    "input_slot": np.int32,
    "output_slot": np.int32,
    "edge_overflow": np.int32,
}


# ---- helpers, reached through the `common` fixture -------------------------------------------


def _float_keys(keys, i_inf):
    """Re-spell ``int32``/``I_INF`` keys as the alternate ``float32``/``NaN`` keys.

    The kernel dispatches on the static key dtype, so both spellings must mean the
    same thing; this is the single conversion site between them.
    """
    keys = np.asarray(keys, dtype=np.int32)
    converted = keys.astype(np.float32)
    converted[keys == i_inf] = np.nan
    return converted


def _key_span(payload, i_inf):
    """``S``: the smallest dense key bound the payload needs (``max(node key) + 1``)."""
    keys = np.asarray(payload.node_keys)
    return int(keys[keys != i_inf].max()) + 1


def _schedule(
    common,
    payload,
    num_inputs=NUM_INPUTS,
    num_outputs=NUM_OUTPUTS,
    *,
    max_in_degree=None,
    key_span=None,
    float_keys=False,
    output_keys=None,
):
    """Derive one payload's schedule through the shared runner and return it as numpy.

    Args:
        common: The ``tensorneat.common`` package fixture.
        payload: One ``support.payloads.Payload`` (a whole population).
        num_inputs, num_outputs: The genome configuration the payload was built
            for; the key vectors are read off the payload's documented rows.
        max_in_degree: ``K``; defaults to ``max_nodes``, the largest in-degree a
            population can express, so nothing overflows unless a test asks.
        key_span: ``S``; defaults to ``max(node key) + 1`` on the host.
        float_keys: Pass the keys in the alternate ``float32``/``NaN`` spelling.
        output_keys: Override the output key vector (an ``(num_outputs,)`` int32
            array), used to feed the kernel an output key no node holds.

    The statics are positional-or-keyword and the ``@evoxc.defn`` object is
    directly callable, so they are passed positionally.
    """
    i_inf = common.I_INF
    node_keys = np.asarray(payload.node_keys)
    conn_keys = np.asarray(payload.conn_keys)
    conn_in = conn_keys[..., 0]
    conn_out = conn_keys[..., 1]
    input_keys = node_keys[0, :num_inputs].copy()
    if output_keys is None:
        output_keys = node_keys[0, payload.max_nodes - num_outputs :].copy()
    else:
        output_keys = np.asarray(output_keys, dtype=np.int32).reshape(num_outputs)

    if max_in_degree is None:
        max_in_degree = payload.max_nodes
    if key_span is None:
        key_span = _key_span(payload, i_inf)

    if float_keys:
        node_keys = _float_keys(node_keys, i_inf)
        conn_in = _float_keys(conn_in, i_inf)
        conn_out = _float_keys(conn_out, i_inf)
        input_keys = _float_keys(input_keys, i_inf)
        output_keys = _float_keys(output_keys, i_inf)

    pop_size, max_nodes = node_keys.shape
    return run_defn(
        common.tensorization.derive_schedule,
        node_keys,
        conn_in,
        conn_out,
        input_keys,
        output_keys,
        pop_size,
        max_nodes,
        conn_in.shape[1],
        int(max_in_degree),
        int(key_span),
    )


def _decode(payload, i_inf, member=0):
    """One member's topology as plain Python, read straight off the payload arrays.

    Returns ``(key_to_slot, edges)``: ``key_to_slot`` maps every present node key
    to its row slot (the row index *is* the slot, by the payload's row invariant),
    and ``edges`` lists ``(conn_row, src_key, dst_key)`` for every row whose two
    endpoint keys both name a present node — exactly the documented ``edge_valid``
    predicate — in connection-row order.
    """
    node_keys = np.asarray(payload.node_keys)[member]
    conn_keys = np.asarray(payload.conn_keys)[member]
    key_to_slot = {int(key): row for row, key in enumerate(node_keys) if key != i_inf}
    edges = []
    for row, pair in enumerate(conn_keys):
        src_key, dst_key = int(pair[0]), int(pair[1])
        if src_key == i_inf or dst_key == i_inf:
            continue
        if src_key not in key_to_slot or dst_key not in key_to_slot:
            continue
        edges.append((row, src_key, dst_key))
    return key_to_slot, edges


def _host_topology(payload, i_inf, member=0):
    """One member's topology in *key* space — what ``host_graph`` consumes.

    ``host_graph.decode_graph`` documents a ``float32``/``NaN`` payload, whereas
    the payload here stores integer keys, so the decode is done locally.
    """
    node_keys = np.asarray(payload.node_keys)[member]
    conn_keys = np.asarray(payload.conn_keys)[member]
    nodes = {int(key) for key in node_keys if key != i_inf}
    conns = {(int(pair[0]), int(pair[1])) for pair in conn_keys if pair[0] != i_inf}
    return nodes, conns


def _edge_groups(payload, i_inf, member=0):
    """``{node_slot: [(conn_row, src_slot), ...]}`` per destination, in row order.

    Oracle for the packing rule of ``in_conn``: a node's in-edges are taken in
    connection-row order, so a group's position *is* its in-group rank.
    """
    key_to_slot, edges = _decode(payload, i_inf, member)
    groups = {}
    for row, src_key, dst_key in edges:
        groups.setdefault(key_to_slot[dst_key], []).append((row, key_to_slot[src_key]))
    return groups


def _incoming_windows(payload, i_inf, max_in_degree, member=0):
    """``(in_conn, in_src, edge_overflow)`` of one member, from the decoded payload.

    Oracle: the first ``max_in_degree`` in-edges of each node in connection-row
    order, every further edge counted in ``edge_overflow``, all other cells
    ``I_INF``. Plain Python via :func:`_edge_groups`, never from a schedule.
    """
    num_nodes = int(np.asarray(payload.node_keys).shape[1])
    in_conn = np.full((num_nodes, max_in_degree), i_inf, dtype=np.int32)
    in_src = np.full((num_nodes, max_in_degree), i_inf, dtype=np.int32)
    overflow = 0
    for node, incoming in _edge_groups(payload, i_inf, member).items():
        for column, (row, src_slot) in enumerate(incoming):
            if column < max_in_degree:
                in_conn[node, column] = row
                in_src[node, column] = src_slot
            else:
                overflow += 1
    return in_conn, in_src, overflow


def _input_keys(payload):
    """The input keys of the whole batch: the first ``NUM_INPUTS`` rows (host ints)."""
    return [int(key) for key in np.asarray(payload.node_keys)[0][:NUM_INPUTS]]


# ---- fixtures: the canonical payloads of ../../support ---------------------------------------


@pytest.fixture(scope="module")
def tiny(payloads):
    """A two-in/one-hidden/one-out chain built row by row (keys ``0,1,2,3``).

    Topology ``0 → 2``, ``1 → 2``, ``2 → 3``; its schedule is fully
    hand-enumerable — levels ``[0, 0, 1, 2]``, node ``2``'s window
    ``((row 0, src 0), (row 1, src 1))``, node ``3``'s ``((row 2, src 2),)``.
    """
    payload = payloads.new_payload(1, 4, 3)
    for row, key in enumerate((0, 1, 2, 3)):
        payloads.set_node(payload, 0, row, key)
    payloads.set_conn(payload, 0, 0, 0, 2)
    payloads.set_conn(payload, 0, 1, 1, 2)
    payloads.set_conn(payload, 0, 2, 2, 3)
    return payload


@pytest.fixture(scope="module")
def chain_payload(payloads):
    """The canonical dense chain (a single path from an input to the output)."""
    return payloads.chain_payload()


@pytest.fixture(scope="module")
def skip_payload(payloads):
    """The canonical topology with a node bypassed by its consumers."""
    return payloads.skip_payload()


@pytest.fixture(scope="module")
def isolated_payload(payloads):
    """The canonical topology with a node that has no edge at all."""
    return payloads.isolated_payload()


@pytest.fixture(scope="module")
def ring_payload(payloads):
    """The canonical recurrent ring (deliberately cyclic)."""
    return payloads.ring_payload()


@pytest.fixture(scope="module")
def sparse_payload(payloads):
    """A reachable chain with real padding rows *and* unclaimed key columns.

    Rows ``0, 1, 2, 7`` are present (keys equal their row index), rows ``3..6``
    are the ``I_INF``/``NaN`` padding rows, and keys ``3..6`` sit inside
    ``[0, key_span)`` yet belong to no node. Topology: ``0 → 2``, ``1 → 2``, ``2 → 7``.
    """
    payload = payloads.new_payload(1, 8, 3)
    for row, key in ((0, 0), (1, 1), (2, 2), (7, 7)):
        payloads.set_node(payload, 0, row, key)
    payloads.set_conn(payload, 0, 0, 0, 2)
    payloads.set_conn(payload, 0, 1, 1, 2)
    payloads.set_conn(payload, 0, 2, 2, 7)
    return payload


@pytest.fixture(scope="module")
def swapped_payload(payloads):
    """A chain whose two in-edges of node ``2`` are *not* in source order.

    Rows are ``(row 0: 1 → 2)``, ``(row 1: 0 → 2)``, ``(row 2: 2 → 7)``, so node
    ``2``'s window is ``((row 0, src 1), (row 1, src 0))``: a window ordered by
    source instead of by connection row reads it backwards.
    """
    payload = payloads.new_payload(1, 8, 3)
    for row, key in ((0, 0), (1, 1), (2, 2), (7, 7)):
        payloads.set_node(payload, 0, row, key)
    payloads.set_conn(payload, 0, 0, 1, 2)
    payloads.set_conn(payload, 0, 1, 0, 2)
    payloads.set_conn(payload, 0, 2, 2, 7)
    return payload


# Derived schedules: one build per distinct (topology, K, key spelling). Read-only;
# the tampering tests copy the field they edit.
@pytest.fixture(scope="module")
def tiny_schedule(common, tiny):
    """The tiny chain's schedule (``K = max_nodes``)."""
    return _schedule(common, tiny)


@pytest.fixture(scope="module")
def chain_schedule(common, chain_payload):
    """The dense chain's schedule, ``max_in_degree = max_nodes`` (nothing dropped)."""
    return _schedule(common, chain_payload)


@pytest.fixture(scope="module")
def chain_schedule_short_window(common, chain_payload):
    """The dense chain's schedule with ``max_in_degree = 1`` — one edge dropped."""
    return _schedule(common, chain_payload, max_in_degree=1)


@pytest.fixture(scope="module")
def skip_schedule(common, skip_payload):
    """The bypass topology's schedule, ``max_in_degree = max_nodes``."""
    return _schedule(common, skip_payload)


@pytest.fixture(scope="module")
def skip_schedule_short_window(common, skip_payload):
    """The bypass topology's schedule with ``max_in_degree = 1``."""
    return _schedule(common, skip_payload, max_in_degree=1)


@pytest.fixture(scope="module")
def isolated_schedule(common, isolated_payload):
    """The isolated-node topology's schedule."""
    return _schedule(common, isolated_payload)


@pytest.fixture(scope="module")
def ring_schedule(common, ring_payload):
    """The recurrent ring's schedule (cyclic rows carry ``I_INF`` levels)."""
    return _schedule(common, ring_payload)


@pytest.fixture(scope="module")
def sparse_schedule(common, sparse_payload):
    """The padded-chain schedule (rows ``3..6`` absent)."""
    return _schedule(common, sparse_payload)


@pytest.fixture(scope="module")
def swapped_schedule(common, swapped_payload):
    """The out-of-source-order chain's schedule, ``max_in_degree = max_nodes``."""
    return _schedule(common, swapped_payload)


@pytest.fixture(scope="module")
def swapped_schedule_short_window(common, swapped_payload):
    """The out-of-source-order chain's schedule with ``max_in_degree = 1``."""
    return _schedule(common, swapped_payload, max_in_degree=1)


# ---- shape contract --------------------------------------------------------------------------


def test_transformed_fields_have_the_documented_shapes_and_dtypes(
    common, chain_payload, chain_schedule
):
    """Invariant: every field is batch-first with the documented shape and dtype.

    Oracle: the shapes spelled out from the payload alone — ``(P, C)`` for
    ``src``/``dst``/``edge_valid``, ``(P, N, K)`` for the window tables, ``(P, N)``
    for the per-node fields, ``(P, I)``/``(P, O)`` for the slots and ``(P,)`` for
    ``edge_overflow`` — with ``int32`` slot fields and ``bool`` masks. A Python
    static leaking into the container would be a violation too.
    """
    schedule = chain_schedule
    pop_size = chain_payload.pop_size
    num_nodes = chain_payload.max_nodes
    num_conns = chain_payload.max_conns
    expected_shapes = {
        "src": (pop_size, num_conns),
        "dst": (pop_size, num_conns),
        "edge_valid": (pop_size, num_conns),
        "in_conn": (pop_size, num_nodes, num_nodes),  # K defaults to max_nodes
        "in_src": (pop_size, num_nodes, num_nodes),
        "in_valid": (pop_size, num_nodes, num_nodes),
        "node_level": (pop_size, num_nodes),
        "node_exists": (pop_size, num_nodes),
        "topo_key": (pop_size, num_nodes),
        "input_slot": (pop_size, NUM_INPUTS),
        "output_slot": (pop_size, NUM_OUTPUTS),
        "edge_overflow": (pop_size,),
    }

    assert isinstance(schedule, common.Transformed)
    assert tuple(common.Transformed._fields) == DOCUMENTED_FIELDS
    for name in DOCUMENTED_FIELDS:
        value = getattr(schedule, name)
        assert isinstance(value, np.ndarray), f"{name} is not a host array"
        assert value.shape == expected_shapes[name], name
        assert value.dtype == np.dtype(DOCUMENTED_DTYPES[name]), name


def test_level_and_overflow_are_compatibility_aliases(common, chain_schedule):
    """Invariant: ``level``/``overflow`` are read-only properties, not fields.

    Oracle: the documented class surface — twelve stored fields plus ``level`` →
    ``node_level`` and ``overflow`` → ``edge_overflow > 0`` for the pre-rewrite
    host callers; a thirteenth field would change the pytree the Pipeline caches.
    """
    schedule = chain_schedule
    assert isinstance(common.Transformed.level, property)
    assert isinstance(common.Transformed.overflow, property)
    assert "level" not in common.Transformed._fields
    assert "overflow" not in common.Transformed._fields
    assert schedule.level is schedule.node_level
    assert np.array_equal(schedule.overflow, schedule.edge_overflow > 0)
    assert schedule.overflow.shape == schedule.edge_overflow.shape


# ---- hand-enumerated schedule of the tiny chain ----------------------------------------------


def test_tiny_chain_levels_are_hand_enumerated(common, tiny, tiny_schedule):
    """Invariant: the tiny chain's levels are literally ``[0, 0, 1, 2]``.

    Oracle: hand enumeration of ``0→2``, ``1→2``, ``2→3`` — the inputs are level
    ``0``, the hidden node one edge below them, the output two.
    """
    assert np.array_equal(tiny_schedule.node_level, np.array([[0, 0, 1, 2]], dtype=np.int32))
    assert np.array_equal(tiny_schedule.node_exists, np.ones((1, 4), dtype=np.bool_))


def test_tiny_chain_incoming_windows_are_hand_enumerated(common, tiny, tiny_schedule):
    """Invariant: the windows are literally ``{2: [(0, 0), (1, 1)], 3: [(2, 2)]}`` (row, src).

    Oracle: hand enumeration of the same three connections; column order is
    connection-row order, and every other column is ``I_INF`` in both tables.
    """
    i_inf = common.I_INF
    in_conn = np.full((4, 4), i_inf, dtype=np.int32)
    in_conn[2, :2] = [0, 1]
    in_conn[3, 0] = 2
    in_src = np.full((4, 4), i_inf, dtype=np.int32)
    in_src[2, :2] = [0, 1]
    in_src[3, 0] = 2

    assert np.array_equal(tiny_schedule.in_conn, in_conn[None, :, :])
    assert np.array_equal(tiny_schedule.in_src, in_src[None, :, :])
    assert np.array_equal(tiny_schedule.in_valid, (in_conn != i_inf)[None, :, :])
    assert np.array_equal(tiny_schedule.edge_overflow, np.zeros(1, dtype=np.int32))


def test_tiny_chain_topology_key_and_slots_are_hand_enumerated(common, tiny, tiny_schedule):
    """Invariant: ``topo_key == level*N + slot``, ``input_slot == [0, 1]``, ``output_slot == [3]``.

    Oracle: hand enumeration (``N == 4``: ``[0*4+0, 0*4+1, 1*4+2, 2*4+3]``); the
    slot vectors follow ``input_keys``/``output_keys`` order.
    """
    assert np.array_equal(tiny_schedule.topo_key, np.array([[0, 1, 6, 11]], dtype=np.int32))
    assert np.array_equal(tiny_schedule.input_slot, np.array([[0, 1]], dtype=np.int32))
    assert np.array_equal(tiny_schedule.output_slot, np.array([[3]], dtype=np.int32))


# ---- levels ----------------------------------------------------------------------------------


def test_input_nodes_are_level_zero(common, chain_schedule):
    """Invariant: every row named by ``input_keys`` has ``level == 0``.

    Oracle: the documented pinning of the inputs to level ``0`` (the relaxation's
    initial values, not a computed depth), on the rows ``0..num_inputs-1`` the
    payload's row invariant fixes, plus the slot vector naming those rows.
    """
    schedule = chain_schedule
    num_inputs = schedule.input_slot.shape[1]
    expected = np.zeros((schedule.node_level.shape[0], num_inputs), dtype=np.int32)

    assert np.array_equal(schedule.node_level[:, :num_inputs], expected)
    assert np.array_equal(
        schedule.input_slot, np.broadcast_to(np.arange(num_inputs), schedule.input_slot.shape)
    )


def test_levels_obey_the_longest_path_recursion(common, chain_payload, chain_schedule, oracle):
    """Invariant: ``level[dst] == 1 + max(level[src] for every edge into dst)``.

    Oracle: three statements of one fact — the per-node recursion evaluated from
    the decoded edge list (:func:`_decode`), the "every edge goes strictly
    upwards" consequence that makes the level sweep a topological order, and the
    reference derivation ``support.oracle.schedule``. On a DAG with pinned inputs
    the recursion has one solution, so the three must agree.
    """
    schedule = chain_schedule
    i_inf = common.I_INF
    member = 0
    key_to_slot, edges = _decode(chain_payload, i_inf)

    predecessors = {}
    for _row, src_key, dst_key in edges:
        src, dst = key_to_slot[src_key], key_to_slot[dst_key]
        assert int(schedule.node_level[member, src]) < int(schedule.node_level[member, dst])
        predecessors.setdefault(dst, []).append(src)

    for node, sources in predecessors.items():
        expected = 1 + max(int(schedule.node_level[member, src]) for src in sources)
        assert int(schedule.node_level[member, node]) == expected

    node_keys = np.asarray(chain_payload.node_keys)
    reference = oracle.schedule(
        node_keys,
        np.asarray(chain_payload.conn_keys)[..., 0],
        np.asarray(chain_payload.conn_keys)[..., 1],
        node_keys[0, :NUM_INPUTS].copy(),
        node_keys[0, -NUM_OUTPUTS:].copy(),
        max_in_degree=chain_payload.max_nodes,
        key_span=_key_span(chain_payload, i_inf),
    )
    assert np.array_equal(schedule.node_level, reference.node_level)


def test_absent_padding_rows_get_level_i_inf(common, sparse_payload, sparse_schedule):
    """Invariant: a padding row has ``level == I_INF`` and ``node_exists == False``.

    Oracle: the documented "``level`` is ``I_INF`` iff node ``i`` is absent", with
    the padding rows read off the payload's key column. The sparse fixture's
    present rows are all reachable, so their levels stay finite — otherwise
    "absent" and "unreachable" would be indistinguishable here.
    """
    absent = np.asarray(sparse_payload.node_keys) == common.I_INF
    schedule = sparse_schedule

    assert absent.any(), "the sparse fixture must carry padding rows"
    assert np.array_equal(schedule.node_exists, ~absent)
    assert np.all(schedule.node_level[absent] == common.I_INF)
    assert np.all(schedule.topo_key[absent] == common.I_INF)
    assert np.all(schedule.node_level[~absent] != common.I_INF)


def test_isolated_node_gets_level_i_inf(common, isolated_payload, isolated_schedule):
    """Invariant: a node with no incoming edge is unreachable, hence ``level == I_INF``.

    Oracle: the documented "unreachable from the inputs" case. The node is
    *present* but can never be relaxed, so its level is the sentinel — not ``0``,
    which would make it look like an input. Its window is empty, and the inputs
    next to it are still pinned at ``0``.
    """
    lone_slot = NUM_INPUTS  # the fixture puts its unconnected node on row num_inputs
    schedule = isolated_schedule

    assert np.asarray(isolated_payload.node_keys)[0, lone_slot] != common.I_INF
    assert schedule.node_exists[0, lone_slot]
    assert schedule.node_level[0, lone_slot] == common.I_INF
    assert not schedule.in_valid[0, lone_slot].any()
    assert np.all(schedule.node_level[0, :NUM_INPUTS] == 0)


def test_cyclic_nodes_get_level_i_inf_and_the_host_walk_raises(
    common, ring_payload, ring_schedule
):
    """Invariant: cyclic nodes carry ``I_INF``, where ``host_graph.node_depth`` raises.

    Oracle: two statements of one fact — the schedule reports an unrelaxable node
    as data (``I_INF``, the documented divergence from a raise), while the host
    walk raises ``GraphError`` for a genome that is no longer feedforward. A
    recurrent genome therefore has *no* depth agreement.
    """
    nodes, conns = _host_topology(ring_payload, common.I_INF)
    cyclic_slots = [2, 3, 7]  # rows 2 and 3 feed each other; output row 7 reads row 3
    schedule = ring_schedule

    assert issubclass(common.GraphError, common.TensorNEATError)
    with pytest.raises(common.GraphError):
        common.host_graph.node_depth(nodes, conns, _input_keys(ring_payload))

    assert np.all(schedule.node_level[0, cyclic_slots] == common.I_INF)
    assert np.all(schedule.topo_key[0, cyclic_slots] == common.I_INF)
    assert np.all(schedule.node_level[0, :NUM_INPUTS] == 0)


def test_level_agrees_with_host_node_depth_on_a_feedforward_genome(
    common, chain_payload, skip_payload, chain_schedule, skip_schedule
):
    """Invariant: for a feedforward genome ``{key: level}`` equals ``host_graph.node_depth``.

    Oracle: the host longest-path walk over the same decoded topology, on two
    shapes with different depths (a chain and a bypass), so an off-by-one or a
    "shortest path" implementation cannot pass.
    """
    i_inf = common.I_INF
    for payload, schedule in ((chain_payload, chain_schedule), (skip_payload, skip_schedule)):
        nodes, conns = _host_topology(payload, i_inf)
        expected = common.host_graph.node_depth(nodes, conns, _input_keys(payload))
        key_to_slot, _edges = _decode(payload, i_inf)
        got = {int(key): int(schedule.node_level[0, slot]) for key, slot in key_to_slot.items()}
        assert set(got) == nodes
        assert got == expected


# ---- incoming-edge windows -------------------------------------------------------------------


def test_incoming_windows_are_dense_and_in_connection_row_order(
    common, chain_payload, chain_schedule, swapped_payload, swapped_schedule
):
    """Invariant: node ``i``'s columns ``0..d-1`` hold its ``d`` incoming rows, ascending.

    Oracle: the decoded connections grouped by destination and taken in
    connection-row order (:func:`_edge_groups`) — the documented "dense prefix +
    in-group rank" rule. The swapped fixture puts the higher-numbered source on
    the lower-numbered row, so a window ordered by source is caught explicitly.
    """
    i_inf = common.I_INF
    for payload, schedule in (
        (chain_payload, chain_schedule),
        (swapped_payload, swapped_schedule),
    ):
        expected_conn, expected_src, _overflow = _incoming_windows(
            payload, i_inf, payload.max_nodes
        )
        assert np.array_equal(schedule.in_conn, expected_conn[None, :, :])
        assert np.array_equal(schedule.in_src, expected_src[None, :, :])

    # rows are (0: 1→2), (1: 0→2), (2: 2→7): node 2's window is row order, not
    # source order, so in_src reads (1, 0) and not (0, 1).
    assert np.array_equal(swapped_schedule.in_conn[0, 2, :2], np.array([0, 1], np.int32))
    assert np.array_equal(swapped_schedule.in_src[0, 2, :2], np.array([1, 0], np.int32))


def test_incoming_windows_carry_the_source_slot_of_the_same_edge(
    common, chain_payload, chain_schedule, swapped_payload, swapped_schedule
):
    """Invariant: ``in_src[p, i, k]`` is the source slot of the edge ``in_conn[p, i, k]`` names.

    Oracle: the source *key* of that connection row, resolved to its slot by the
    payload's own key→slot map (:func:`_decode`) — not ``src[in_conn]`` read back
    from the schedule. The identity is then checked against ``src``, the
    cross-reference the forward pass relies on when it gathers source values with
    ``in_src`` but connection attributes with ``in_conn``.
    """
    i_inf = common.I_INF
    member = 0
    for payload, schedule in (
        (chain_payload, chain_schedule),
        (swapped_payload, swapped_schedule),
    ):
        key_to_slot, edges = _decode(payload, i_inf)
        src_by_row = np.full(payload.max_conns, i_inf, dtype=np.int32)
        for row, src_key, _dst_key in edges:
            src_by_row[row] = key_to_slot[src_key]

        valid = schedule.in_valid[member]
        live_rows = np.where(valid, schedule.in_conn[member], 0)  # sanitized index
        expected = np.where(valid, src_by_row[live_rows], i_inf)
        assert np.array_equal(schedule.in_src[member], expected)

        edge_valid = schedule.edge_valid[member]
        assert np.array_equal(schedule.src[member][edge_valid], src_by_row[edge_valid])
        assert np.array_equal(
            schedule.in_src[member], np.where(valid, schedule.src[member][live_rows], i_inf)
        )


def test_padded_columns_never_leak_into_values(
    common, chain_payload, skip_payload, chain_schedule, skip_schedule
):
    """Invariant: ``in_conn`` is ``I_INF`` exactly where ``in_src`` is, padding only above ``d``.

    Oracle: the two invariances of the ``Transformed`` docstring — identical
    sentinel masks in ``in_conn``/``in_src``/``in_valid``, and per node "the dense
    prefix is ``d`` long" with ``d`` counted from the decoded edges. This is what
    lets a forward pass gather with ``I_INF`` padding: a padded column carrying a
    real value (or an ``I_INF`` hiding a real edge) reads somebody else's data.
    """
    i_inf = common.I_INF
    for payload, schedule in ((chain_payload, chain_schedule), (skip_payload, skip_schedule)):
        expected_conn, expected_src, _overflow = _incoming_windows(
            payload, i_inf, payload.max_nodes
        )
        assert np.array_equal(schedule.in_conn, expected_conn[None, :, :])
        assert np.array_equal(schedule.in_src, expected_src[None, :, :])
        assert np.array_equal(schedule.in_valid, schedule.in_conn != i_inf)
        assert np.array_equal(schedule.in_valid, schedule.in_src != i_inf)

        for node, incoming in _edge_groups(payload, i_inf).items():
            depth = len(incoming)
            assert schedule.in_valid[0, node, :depth].all()
            assert not schedule.in_valid[0, node, depth:].any()


# ---- edge_overflow ---------------------------------------------------------------------------


def test_edge_overflow_is_zero_when_the_bound_is_large_enough(
    common, chain_payload, chain_schedule
):
    """Invariant: ``edge_overflow == 0`` whenever ``max_in_degree`` covers every in-degree.

    Oracle: the in-degrees counted from the payload's edge list, compared with the
    static ``K`` (= ``max_nodes`` here); nothing may be reported as dropped when
    nothing was. The host-side ``overflow`` predicate must agree.
    """
    i_inf = common.I_INF
    degrees = {
        node: len(incoming) for node, incoming in _edge_groups(chain_payload, i_inf).items()
    }
    assert max(degrees.values()) <= chain_payload.max_nodes
    assert np.array_equal(
        chain_schedule.edge_overflow, np.zeros(chain_payload.pop_size, dtype=np.int32)
    )
    assert not chain_schedule.overflow.any()


def test_edge_overflow_counts_exactly_the_dropped_edges(
    common,
    chain_payload,
    swapped_payload,
    chain_schedule_short_window,
    swapped_schedule_short_window,
):
    """Invariant: with ``K`` one short of a node's in-degree ``d``, ``edge_overflow == d - K``.

    Oracle: the in-degree counted from the decoded edges — the overflow is exactly
    the edges beyond the bound, and the retained window is the first ``K`` rows in
    connection-row order. ``edge_overflow`` is data, so no exception is expected:
    this is the documented "the host decides to rebuild" path. In the swapped
    fixture the dropped edge is the higher-numbered row whose source is the lower
    key, so "keep the lowest ranks" cannot be mistaken for "keep the lowest sources".
    """
    i_inf = common.I_INF
    for payload, schedule in (
        (chain_payload, chain_schedule_short_window),
        (swapped_payload, swapped_schedule_short_window),
    ):
        expected_conn, expected_src, expected_overflow = _incoming_windows(payload, i_inf, 1)
        degrees = [len(incoming) for incoming in _edge_groups(payload, i_inf).values()]
        assert expected_overflow == sum(max(0, degree - 1) for degree in degrees)

        assert np.array_equal(schedule.in_conn, expected_conn[None, :, :])
        assert np.array_equal(schedule.in_src, expected_src[None, :, :])
        assert np.array_equal(
            schedule.edge_overflow, np.array([expected_overflow], dtype=np.int32)
        )
        assert schedule.overflow[0]

    # literal: the swapped fixture's node 2 keeps connection row 0 (source key 1)
    # and loses row 1 (source key 0).
    assert np.array_equal(
        swapped_schedule_short_window.in_conn[0, 2], np.array([0], dtype=np.int32)
    )
    assert np.array_equal(
        swapped_schedule_short_window.in_src[0, 2], np.array([1], dtype=np.int32)
    )
    assert np.array_equal(
        swapped_schedule_short_window.edge_overflow, np.array([1], dtype=np.int32)
    )


# ---- the whole schedule vs the independent numpy oracle --------------------------------------


def test_derive_schedule_matches_the_oracle_field_by_field(
    common,
    chain_payload,
    skip_payload,
    oracle,
    chain_schedule,
    chain_schedule_short_window,
    skip_schedule,
    skip_schedule_short_window,
):
    """Invariant: every ``Transformed`` field equals ``support.oracle.schedule``'s.

    Oracle: the reference derivation (dense key→slot table, in-group rank, counting
    trick, longest-path levels) on the same arrays. All twelve fields are compared,
    dtype included, on two topologies and two window bounds (``K == max_nodes`` and
    the lossy ``K == 1``), so a shape-specific coincidence is unlikely.
    """
    i_inf = common.I_INF
    cases = (
        (chain_payload, chain_schedule, chain_payload.max_nodes),
        (chain_payload, chain_schedule_short_window, 1),
        (skip_payload, skip_schedule, skip_payload.max_nodes),
        (skip_payload, skip_schedule_short_window, 1),
    )
    for payload, schedule, max_in_degree in cases:
        node_keys = np.asarray(payload.node_keys)
        conn_keys = np.asarray(payload.conn_keys)
        reference = oracle.schedule(
            node_keys,
            conn_keys[..., 0],
            conn_keys[..., 1],
            node_keys[0, :NUM_INPUTS].copy(),
            node_keys[0, -NUM_OUTPUTS:].copy(),
            max_in_degree=max_in_degree,
            key_span=_key_span(payload, i_inf),
        )
        for name in DOCUMENTED_FIELDS:
            expected = getattr(reference, name)
            actual = getattr(schedule, name)
            assert actual.dtype == expected.dtype, f"{name} dtype with K={max_in_degree}"
            assert np.array_equal(actual, expected), f"{name} with K={max_in_degree}"


def test_topological_key_is_a_valid_execution_order(common, chain_payload, chain_schedule):
    """Invariant: every valid edge goes strictly upwards, i.e. ``level[src] < level[dst]``.

    Oracle: the documented "executing levels ``0, 1, ...`` in order is a valid
    topological order", over the payload's decoded edges, plus the documented
    ``topo_key == level * N + slot`` identity (what a sweep sorts on) and its
    consequence ``topo_key[src] < topo_key[dst]``.
    """
    i_inf = common.I_INF
    member = 0
    schedule = chain_schedule
    num_nodes = schedule.node_level.shape[1]
    key_to_slot, edges = _decode(chain_payload, i_inf)

    for _row, src_key, dst_key in edges:
        src, dst = key_to_slot[src_key], key_to_slot[dst_key]
        assert int(schedule.node_level[member, src]) < int(schedule.node_level[member, dst])
        assert int(schedule.topo_key[member, src]) < int(schedule.topo_key[member, dst])

    live = schedule.node_level != i_inf
    slots = np.broadcast_to(np.arange(num_nodes), schedule.node_level.shape)
    expected = np.where(live, schedule.node_level * num_nodes + slots, i_inf)
    assert np.array_equal(schedule.topo_key, expected)


# ---- validate_schedule: the host checker -----------------------------------------------------


def test_validate_schedule_accepts_the_kernel_output(
    common,
    chain_schedule,
    skip_schedule,
    isolated_schedule,
    ring_schedule,
    chain_schedule_short_window,
):
    """Invariant: ``validate_schedule`` returns ``None`` for every kernel schedule.

    Oracle: the kernel itself (the acceptance half) plus the documented ``None``
    return — a checker that returned a *repaired* schedule would hide the
    violation it was asked to report. The sample covers ``I_INF`` levels (the
    ring), absent rows (the isolated topology) and a lossy window (``K == 1``),
    because all three are valid schedules.
    """
    for schedule in (
        chain_schedule,
        skip_schedule,
        isolated_schedule,
        ring_schedule,
        chain_schedule_short_window,
    ):
        assert common.tensorization.validate_schedule(schedule) is None


def test_validate_schedule_rejects_in_conn_in_src_disagreement(common, chain_schedule):
    """Invariant: a padded cell in one table and a real index in the other is a ``ConfigError``.

    Oracle: a schedule tampered with ``Transformed._replace`` — one ``in_src``
    padding cell set to a real slot while ``in_conn`` keeps ``I_INF``. The tables
    must agree on every sentinel, so this must raise, naming the invariant.
    """
    schedule = chain_schedule
    assert schedule.in_conn[0, 0, 0] == common.I_INF  # node 0 has no incoming edge

    tampered = schedule.in_src.copy()
    tampered[0, 0, 0] = 0  # a real slot where in_conn says I_INF
    with pytest.raises(common.ConfigError, match="in_conn and in_src"):
        common.tensorization.validate_schedule(schedule._replace(in_src=tampered))


def test_validate_schedule_rejects_i_inf_level_with_a_finite_topo_key(common, sparse_schedule):
    """Invariant: ``level == I_INF`` implies ``topo_key == I_INF``; a finite key raises.

    Oracle: the documented pair ``topo_key = level * N + slot`` plus its sentinel —
    an unavailable node cannot be given a place in the sweep. Tampered by giving
    one absent row ``topo_key == 0``, a place in the very first level.
    """
    schedule = sparse_schedule
    dead_rows = np.flatnonzero(schedule.node_level[0] == common.I_INF)
    assert dead_rows.size, "the sparse fixture must have an I_INF level to tamper with"

    tampered = schedule.topo_key.copy()
    tampered[0, int(dead_rows[0])] = 0
    with pytest.raises(
        common.ConfigError, match="I_INF node_level must pair with an I_INF topo_key"
    ):
        common.tensorization.validate_schedule(schedule._replace(topo_key=tampered))


def test_validate_schedule_rejects_negative_edge_overflow(common, chain_schedule):
    """Invariant: a negative ``edge_overflow`` is a ``ConfigError``.

    Oracle: the field's meaning (a *count* of dropped edges); a negative count is
    impossible for any derivation, so it always means a corrupted or hand-built
    schedule, which must not reach a rebuild decision.
    """
    schedule = chain_schedule
    assert np.all(schedule.edge_overflow >= 0)

    tampered = schedule.edge_overflow.copy()
    tampered[0] = -1
    with pytest.raises(common.ConfigError, match="cannot be negative"):
        common.tensorization.validate_schedule(schedule._replace(edge_overflow=tampered))


def test_validate_schedule_rejects_an_out_of_range_input_slot(common, chain_schedule):
    """Invariant: an ``input_slot`` outside ``[0, N)`` — and not ``I_INF`` — is a ``ConfigError``.

    Oracle: the documented "slots point inside ``[0, N)`` or are the sentinel";
    ``N`` itself is out of range (the first illegal index), the mistake an
    off-by-one derivation makes.
    """
    schedule = chain_schedule
    tampered = schedule.input_slot.copy()
    tampered[0, 0] = schedule.node_level.shape[1]
    with pytest.raises(common.ConfigError, match="input_slot must hold slots"):
        common.tensorization.validate_schedule(schedule._replace(input_slot=tampered))


def _tamper_in_valid(schedule, i_inf):
    """Mark a padded window cell live while ``in_conn`` keeps ``I_INF``."""
    assert not schedule.in_valid[0, 0, 0]
    tampered = schedule.in_valid.copy()
    tampered[0, 0, 0] = True
    return schedule._replace(in_valid=tampered)


def _tamper_live_level_with_i_inf_key(schedule, i_inf):
    """Blank the ``topo_key`` of a node that *does* have a finite level."""
    node = int(np.flatnonzero(schedule.node_level[0] != i_inf)[-1])
    tampered = schedule.topo_key.copy()
    tampered[0, node] = i_inf
    return schedule._replace(topo_key=tampered)


def _tamper_topo_key_formula(schedule, i_inf):
    """Shift one live ``topo_key`` off ``node_level * N + slot``."""
    node = int(np.flatnonzero(schedule.node_level[0] != i_inf)[-1])
    assert schedule.topo_key[0, node] != i_inf
    tampered = schedule.topo_key.copy()
    tampered[0, node] = int(tampered[0, node]) + 1
    return schedule._replace(topo_key=tampered)


def _tamper_field_shape(schedule, i_inf):
    """Give the ``(P, N)`` node level the wrong node count."""
    return schedule._replace(node_level=schedule.node_level[:, :1])


def _tamper_field_dtype(schedule, i_inf):
    """Turn an integer field into a float array."""
    return schedule._replace(edge_overflow=schedule.edge_overflow.astype(np.float64))


@pytest.mark.parametrize(
    "tamper, message",
    [
        (_tamper_in_valid, "in_valid must be exactly in_conn != I_INF"),
        (
            _tamper_live_level_with_i_inf_key,
            "a finite node_level must pair with a finite topo_key",
        ),
        (_tamper_topo_key_formula, "topo_key must equal node_level \\* N \\+ slot"),
        (_tamper_field_shape, "node_level has shape"),
        (_tamper_field_dtype, "edge_overflow must be an integer array"),
    ],
    ids=["in_valid", "live-level-dead-key", "topo-key-formula", "shape", "dtype"],
)
def test_validate_schedule_rejects_each_remaining_tampered_invariant(
    common, chain_schedule, tamper, message
):
    """Invariant: each remaining documented rejection is a ``ConfigError`` naming its rule.

    Oracle: the ``Raises`` list of :func:`validate_schedule` — ``in_valid``
    disagreeing with ``in_conn != I_INF``, a live level paired with a dead
    ``topo_key``, a live ``topo_key`` off ``node_level * N + slot``, a field of the
    wrong shape, and a non-integer field. Each case tampers with one cell/field of
    a kernel-produced schedule, so no other invariant can fire first.
    """
    tampered = tamper(chain_schedule, common.I_INF)
    with pytest.raises(common.ConfigError, match=message):
        common.tensorization.validate_schedule(tampered)


# ---- the two leaf helpers of the derivation --------------------------------------------------


def test_key_slot_table_marks_absent_and_unknown_keys(common, sparse_payload):
    """Invariant: a real key maps to its row slot; absent rows and unknown keys map to ``I_INF``.

    Oracle: the payload's own key column — the expected table is ``{key: row}``
    read straight off ``node_keys``, with ``I_INF`` on every column no node claims.
    The width is ``key_span + 1`` because the trailing column is the write dump
    that keeps an absent row away from slot ``0`` (its *content* is documented as
    irrelevant, so it is deliberately not asserted).
    """
    i_inf = common.I_INF
    key_span = _key_span(sparse_payload, i_inf)
    node_keys = np.asarray(sparse_payload.node_keys)
    table = run_defn(common.tensorization.key_slot_table, node_keys, key_span)

    assert table.dtype == np.int32
    assert table.shape == (sparse_payload.pop_size, key_span + 1)
    expected = np.full((sparse_payload.pop_size, key_span), i_inf, dtype=np.int32)
    for member in range(sparse_payload.pop_size):
        for row, key in enumerate(node_keys[member]):
            if key != i_inf:
                expected[member, int(key)] = row
    assert np.array_equal(table[:, :key_span], expected)

    unclaimed = sorted(set(range(key_span)) - {int(key) for key in node_keys[0] if key != i_inf})
    assert unclaimed, "the sparse fixture must leave key columns unclaimed"
    for key in unclaimed:
        assert np.all(table[:, key] == i_inf)


def test_slot_lookup_maps_nan_and_unknown_keys_to_i_inf(common, sparse_payload):
    """Invariant: ``slot_lookup`` returns ``I_INF`` for ``NaN`` and for a key no node holds.

    Oracle: the table built by ``key_slot_table`` plus a hand-built query vector of
    ``[held key, NaN, key_span (out of range), negative key, unclaimed key]``. The
    sentinel must survive the index sanitizing that precedes the gather: slot ``0``
    is a real node (key ``0``), so a leak from clamping shows up as ``0``.
    """
    i_inf = common.I_INF
    key_span = _key_span(sparse_payload, i_inf)
    node_keys = _float_keys(np.asarray(sparse_payload.node_keys), i_inf)
    table = run_defn(common.tensorization.key_slot_table, node_keys, key_span)

    held_key = float(sparse_payload.max_nodes - 1)  # key 7, held by row 7
    unclaimed_key = 4.0  # inside [0, key_span), held by no node
    queries = np.array([[held_key, np.nan, float(key_span), -1.0, unclaimed_key]], np.float32)
    slots = run_defn(common.tensorization.slot_lookup, table, queries, key_span)

    expected = np.array([[sparse_payload.max_nodes - 1, i_inf, i_inf, i_inf, i_inf]], np.int32)
    assert np.array_equal(slots, expected)
    assert np.all(slots[0, 1:] == i_inf)  # no sanitizing leak into slot 0


def test_missing_output_key_reports_a_miss_slot(common, sparse_payload):
    """Invariant: an output key no node holds yields ``output_slot == I_INF``, not a raise.

    Oracle: the kernel's documented ``Notes`` clause — an input/output key no node
    holds surfaces as an ``I_INF`` slot (a miss) instead of raising, and refusing
    such a payload is host code's job, not the trace's. ``4`` is inside
    ``[0, key_span)`` yet held by no node of the sparse fixture, so the miss cannot
    come from the range test alone; the schedule stays valid either way.
    """
    i_inf = common.I_INF
    held = _schedule(common, sparse_payload)
    assert held.output_slot[0, 0] == sparse_payload.max_nodes - 1  # key 7 → row 7

    missed = _schedule(common, sparse_payload, output_keys=np.array([4], dtype=np.int32))
    assert np.all(missed.output_slot == i_inf)
    assert common.tensorization.validate_schedule(missed) is None


# ---- the two key spellings -------------------------------------------------------------------


def test_float_key_spelling_matches_the_int_key_spelling(
    common, chain_payload, ring_payload, chain_schedule, ring_schedule
):
    """Invariant: ``float32``/``NaN`` keys produce the identical schedule as ``int32``/``I_INF``.

    Oracle: the canonical ``int32`` schedule — the kernel dispatches on the static
    key dtype, so the sentinel may only change how an absent row is *spelled*,
    never what the schedule says. The ring is included because it is the fixture
    with absent rows *and* unused connection rows, i.e. both sentinel paths.
    """
    for payload, int_schedule in ((chain_payload, chain_schedule), (ring_payload, ring_schedule)):
        float_schedule = _schedule(common, payload, float_keys=True)
        for name in DOCUMENTED_FIELDS:
            assert np.array_equal(
                getattr(float_schedule, name), getattr(int_schedule, name)
            ), f"{name} differs between the two key spellings"
