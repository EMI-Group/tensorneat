"""Tensorization: turn a ragged evolved topology into a static execution schedule.

Bottom layer of ``tensorneat.common``: imports
:mod:`~tensorneat.common.tools` and :mod:`~tensorneat.common.errors`, nothing
from ``genome``/``algorithm``/``problem``/``pipeline`` and no ``jax``/``torch``.
It replaces the pre-rewrite ``common/graph.py``, whose topological sort ran over
a dense ``(N, N)`` adjacency matrix at ``max_nodes ** 2`` cost.

A schedule resolves every ragged connection row to ``(src, dst)`` **node slots**
(indices into that genome's own arrays, never innovation numbers), packs each
node's incoming edges into a ``(P, N, K)`` table and gives every node a
``node_level`` (longest-path depth) that orders a forward pass by level — work
proportional to real connections and depth, never to ``max_nodes ** 2``.
:class:`Transformed` holds **arrays only**, so it is a plain pytree that crosses
a graph boundary and is cached per shape signature.

Cost: ``O(P * (S + C log C + N * K * D))`` with ``S = key_span``, ``C =
max_conns``, ``K = max_in_degree`` and ``D`` = network depth, paid once per
generation; only the cycle-detecting fallback of :func:`node_levels` can reach
``2N`` relaxation steps, and only for a population that holds a cycle.

``evoxc`` constraints that shape this module: no ``cumsum`` (a StableHLO export
deferral, hence the ``cc`` counting trick), no scatter-with-reduce, no batched
``take_along_axis`` (hence flat indexing), no ``where`` (``evoxc.select``), and
out-of-range indices raise instead of clamping — every table index is sanitized
*and* masked. Everything here is in-graph: no host round-trip, no ``.numpy()``
and no Python ``if`` on a traced value (only on statics).
"""

from typing import Any, NamedTuple

import evoxc
import evoxc.numpy as enp
import numpy as np

from .errors import ConfigError
from .tools import (
    I_INF,
    KEY_MAX,
    _positive_static,
    _scalar_constant,
    flat_gather_prepared,
    flat_scatter,
    flat_scatter_prepared,
    masked_flat_scatter,
    prepare_flat_index,
    to_numpy,
)


class Transformed(NamedTuple):
    """Static execution schedule of one population; every field is batch-first.

    ``N = max_nodes``, ``C = max_conns``, ``K = max_in_degree``, ``I =
    len(input_keys)`` and ``O = len(output_keys)`` are static at build time.
    ``I_INF`` (``int32`` max) means "no index / unmatched / unavailable"; ``NaN``
    lives in the gene payload, never here. The twelve fields below are the
    contract, ``level`` / ``overflow`` are read-only compatibility properties.

    Invariants: ``in_valid == (in_conn != I_INF)`` marks the same positions in
    ``in_src``, and a node's window columns ``0..d-1`` are dense; every valid row
    appears exactly once as ``in_conn[p, i, k]``/``in_src[p, i, k]`` unless its
    in-group rank reached ``K`` (``edge_overflow``: a lossy schedule the host must
    rebuild); ``node_level`` is ``I_INF`` iff the node is absent, unreachable from
    the inputs, or on / downstream of a live cycle, and it bounds the execution
    loop; the levels ``0, 1, ...`` in order are a valid topological order.
    """

    #: (P, C) int32 — source node slot of each connection row; I_INF = absent row or unmatched key.
    src: Any
    #: (P, C) int32 — destination node slot of each connection row; I_INF likewise.
    dst: Any
    #: (P, C) bool — True iff the row is a real connection AND both endpoint keys matched a node.
    edge_valid: Any
    #: (P, N, K) int32 — conn row index of the k-th incoming edge of node i; I_INF padding.
    in_conn: Any
    #: (P, N, K) int32 — source slot of that same incoming edge; I_INF in the same positions as in_conn.
    in_src: Any
    #: (P, N, K) bool — that window slot holds a live edge, i.e. ``in_conn != I_INF``.
    in_valid: Any
    #: (P, N) int32 — longest-path level (inputs = 0); I_INF = absent, unreachable or cyclic.
    node_level: Any
    #: (P, N) bool — the node row is present (its key column is not the padding sentinel).
    node_exists: Any
    #: (P, N) int32 — node_level * N + slot: a cheap total order for a topological sweep; I_INF where node_level is.
    topo_key: Any
    #: (P, I) int32 — node slot carrying input j, in the order of input_keys; I_INF if unmatched.
    input_slot: Any
    #: (P, O) int32 — node slot carrying output j, in the order of output_keys; I_INF if unmatched.
    output_slot: Any
    #: (P,) int32 — number of valid edges dropped because their in-group rank reached K.
    edge_overflow: Any

    @property
    def level(self) -> Any:
        """Compatibility alias of :attr:`node_level` (numpy oracle and pre-rewrite host callers)."""
        return self.node_level

    @property
    def overflow(self) -> Any:
        """Compatibility alias: ``edge_overflow > 0``, the "schedule is lossy" predicate."""
        return self.edge_overflow > 0


# ============================================================ trace-time helpers
# Every static count here is validated by ``tools._positive_static`` (imported
# above), and every rank-0 constant is built by ``tools._scalar_constant``:
# one owner per rule, so no helper below duplicates them.


def _check_shape(tensor: Any, expected: tuple, name: str) -> None:
    """Raise :class:`ConfigError` unless ``tensor``'s static shape is ``expected``."""
    shape = tuple(int(size) for size in tensor.shape)
    if shape != tuple(expected):
        raise ConfigError(f"{name} must have shape {tuple(expected)}, got {shape}")


def _iinf_like(tensor: Any) -> Any:
    """``I_INF`` broadcast to ``tensor``'s shape and integer dtype (never used on floats)."""
    return enp.full(tuple(tensor.shape), I_INF, dtype=tensor.dtype)


def _row_broadcast(vector: Any, pop_size: int) -> Any:
    """Broadcast a shared ``(L,)`` key vector to ``(P, L)`` (key vectors carry no batch axis)."""
    width = int(vector.shape[0])
    return enp.broadcast_to(evoxc.reshape(vector, (1, width)), (pop_size, width))


def _node_slots(pop_size: int, count: int) -> Any:
    """``(P, count)`` ``int32`` tensor holding ``[0, 1, ..., count - 1]`` on every row."""
    row = evoxc.reshape(enp.arange(0, count, dtype=evoxc.int32), (1, count))
    return enp.broadcast_to(row, (pop_size, count))


def _key_present(keys: Any) -> Any:
    """Mask of the entries holding a real key rather than the padding sentinel.

    ``int32`` keys compare against ``I_INF``; ``float32`` keys are "not ``NaN`` and
    below ``KEY_MAX``" (evoxc has no ``isfinite``). A trace-time branch on the static
    dtype; a negative or out-of-span key is dropped later by :func:`_key_indices`,
    so both spellings agree on every real key.
    """
    if keys.dtype.kind in ("i", "u"):
        return enp.not_equal(keys, _iinf_like(keys))
    return enp.logical_and(
        enp.logical_not(evoxc.isnan(keys)),
        enp.less(keys, enp.full(tuple(keys.shape), KEY_MAX, dtype=keys.dtype)),
    )


def _narrow_int(limit):
    """Narrowest integer dtype that holds ``limit``, for a radix-sortable key.

    evoxc has no unsigned dtypes, so the split is ``int8`` (``limit < 2**7``),
    ``int16`` (``< 2**15``) and ``int32`` — numpy counting-sorts a narrow integer
    key instead of comparison-sorting it.
    """
    if limit >= I_INF:
        raise ConfigError(
            f"static sort-key bound {limit} does not fit an int32 carrier "
            f"(I_INF = {I_INF})"
        )
    if limit <= 127:
        return evoxc.int8
    if limit <= 32767:
        return evoxc.int16
    return evoxc.int32


def _key_indices(keys: Any, key_span: int) -> tuple:
    """Sanitized ``(P, L)`` ``int32`` table indices plus the "the key is real" mask.

    ``NaN``/``I_INF`` keys and keys outside ``[0, key_span)`` map to ``0`` before the
    gather (evoxc raises on out-of-range indices) and report ``False`` in the mask,
    so the caller never reads a sanitized cell by accident.
    """
    zero = enp.zeros(tuple(keys.shape), dtype=keys.dtype)
    if keys.dtype.kind in ("i", "u"):
        in_range = enp.less(
            evoxc.cast(keys, evoxc.uint32),
            _scalar_constant(key_span, evoxc.uint32),
        )
        return evoxc.cast(evoxc.select(in_range, keys, zero), evoxc.int32), in_range
    in_range = enp.logical_and(
        _key_present(keys),
        enp.logical_and(
            enp.greater_equal(keys, zero),
            enp.less(keys, enp.full(tuple(keys.shape), key_span, dtype=keys.dtype)),
        ),
    )
    return evoxc.cast(evoxc.select(in_range, keys, zero), evoxc.int32), in_range


# ============================================================== the composed graph
@evoxc.defn
def derive_schedule(
    node_keys,
    conn_in,
    conn_out,
    input_keys,
    output_keys,
    pop_size,
    max_nodes,
    max_conns,
    max_in_degree,
    key_span,
    acyclic=False,
):
    """Derive the execution schedule of one whole population (one evoxc graph).

    Args:
        node_keys: ``(P, max_nodes)`` node **key column**, ``int32`` with ``I_INF``
            padding or ``float32`` with ``NaN`` padding (dispatch on the static
            dtype, see :func:`_key_present`); conn_in/conn_out: ``(P, max_conns)``
            source/destination keys, same dtype and padding; input_keys ``(I,)`` and
            output_keys ``(O,)``: shared key vectors, no batch axis.
        pop_size (int): ``P``; max_nodes (int): ``N``; max_conns (int): ``C``;
            max_in_degree (int): ``K``; key_span (int): ``S``, the magnitude bound of
            the ``(P, S + 1)`` key→slot table. All trace-time statics, positional-
            or-keyword because ``evoxc.build`` mirrors a positional call.
        acyclic (bool): ``False`` (default) detects live cycles — a genome on or
            downstream of one gets ``I_INF`` levels — for the price of the level-only
            relaxation plus an in-graph test, the cycle-detecting fallback running
            in ``evoxc.cond`` only where a cycle exists. ``True`` asserts a cycle-free
            population and drops both, so a cyclic genome then gets a saturated level
            instead of ``I_INF`` (feed-forward may pass ``True``, recurrent may not).

    Returns:
        Transformed: the schedule described by :class:`Transformed` — arrays only,
        so the pytree can cross a graph boundary and be cached per shape.

    Raises:
        tensorneat.common.errors.ConfigError: At trace time, for a non-positive or
            shape-inconsistent static, or for key arrays of differing dtypes.

    Notes:
        One helper per step, each individually testable: :func:`key_slot_table`,
        :func:`slot_lookup` (an unmatched endpoint is ``I_INF`` and invalidates the
        row), :func:`edge_order` (in-edge rank and the ``cumsum``-free per-node
        prefix count from **one** stable sort), :func:`incoming_table` (``col = rank
        - start[dst]``, after which ``in_valid`` is free) and :func:`node_levels`.
        A key outside ``[0, key_span)`` and an input/output key no node holds cannot
        raise in-graph, so both surface as ``I_INF`` slots (a miss): the host that
        owns the payload refuses such a payload, this kernel reports, never guesses.
    """
    pop_size = _positive_static(pop_size, "pop_size")
    max_nodes = _positive_static(max_nodes, "max_nodes")
    max_conns = _positive_static(max_conns, "max_conns")
    max_in_degree = _positive_static(max_in_degree, "max_in_degree")
    key_span = _positive_static(key_span, "key_span")
    _check_shape(node_keys, (pop_size, max_nodes), "node_keys")
    _check_shape(conn_in, (pop_size, max_conns), "conn_in")
    _check_shape(conn_out, (pop_size, max_conns), "conn_out")
    for name, keys in (("num_inputs", input_keys), ("num_outputs", output_keys)):
        if int(keys.shape[0]) > max_nodes:
            raise ConfigError(f"{name} {int(keys.shape[0])} exceeds max_nodes {max_nodes}")
    for name, tensor in (
        ("conn_in", conn_in),
        ("conn_out", conn_out),
        ("input_keys", input_keys),
        ("output_keys", output_keys),
    ):
        if tensor.dtype != node_keys.dtype:
            raise ConfigError(
                f"{name} dtype {tensor.dtype} must match node_keys dtype {node_keys.dtype}"
            )

    table = key_slot_table(node_keys, key_span)
    src = _slot_of(table, conn_in, key_span)
    dst = _slot_of(table, conn_out, key_span)
    edge_valid = enp.logical_and(
        enp.not_equal(src, _iinf_like(src)), enp.not_equal(dst, _iinf_like(dst))
    )
    input_slot = _slot_of(table, _row_broadcast(input_keys, pop_size), key_span)
    output_slot = _slot_of(table, _row_broadcast(output_keys, pop_size), key_span)
    rank, cc = edge_order(dst, edge_valid, max_nodes, max_conns)
    in_conn, in_src, edge_overflow = incoming_table(
        rank, cc, dst, src, edge_valid, max_nodes, max_in_degree
    )
    in_valid = enp.not_equal(in_conn, _iinf_like(in_conn))
    node_level = node_levels(
        in_conn, in_src, input_slot, max_nodes, acyclic, (src, dst, edge_valid)
    )
    node_exists = _key_present(node_keys)
    slots = _node_slots(pop_size, max_nodes)
    has_level = enp.not_equal(node_level, _iinf_like(node_level))
    ordered = evoxc.add(
        evoxc.multiply(evoxc.select(has_level, node_level, enp.zeros(tuple(node_level.shape), dtype=evoxc.int32)), _scalar_constant(max_nodes, evoxc.int32)),
        slots,
    )
    topo_key = evoxc.select(has_level, ordered, _iinf_like(ordered))
    return Transformed(
        src=src,
        dst=dst,
        edge_valid=edge_valid,
        in_conn=in_conn,
        in_src=in_src,
        in_valid=in_valid,
        node_level=node_level,
        node_exists=node_exists,
        topo_key=topo_key,
        input_slot=input_slot,
        output_slot=output_slot,
        edge_overflow=edge_overflow,
    )


# ===================================================================== the steps
def key_slot_table(node_keys, key_span):
    """Build the dense per-genome key→node-slot table.

    Args:
        node_keys: ``(P, N)`` node key column, ``int32`` with ``I_INF`` padding or
            ``float32`` with ``NaN`` padding; real keys are integer-valued in
            ``[0, key_span)``.
        key_span (int): ``S``, the static key-magnitude bound.

    Returns:
        ``(P, S + 1)`` ``int32``: ``table[p, k]`` is the slot of genome ``p``'s node
        with key ``k``, ``I_INF`` when no node holds it (a key outside ``[0, S)``, or
        under the float spelling at or above ``2**24``/``tools.KEY_MAX``, is a miss).
        Column ``S`` is the **dump slot** that makes one unmasked scatter safe.
    """
    key_span = _positive_static(key_span, "key_span")
    pop_size, max_nodes = node_keys.shape
    indices, in_range = _key_indices(node_keys, key_span)
    table = enp.full((pop_size, key_span + 1), I_INF, dtype=evoxc.int32)
    slots = _node_slots(int(pop_size), int(max_nodes))
    return masked_flat_scatter(table, indices, slots, in_range, trusted=True)


def slot_lookup(table, keys, key_span):
    """Look up node slots for a batch of keys, yielding ``I_INF`` on a miss.

    Args:
        table: ``(P, S + 1)`` ``int32`` output of :func:`key_slot_table`.
        keys: ``(P, L)`` keys to resolve — connection endpoints, or a broadcast
            shared key vector — same dtype as the table; ``NaN``, ``I_INF``, unknown
            keys and keys outside ``[0, key_span)`` are all misses.
        key_span (int): ``S``, matching the table.

    Returns:
        ``(P, L)`` ``int32`` slots in ``[0, N)``, ``I_INF`` on a miss.

    Notes:
        Indices are made safe *before* the gather (sentinel → ``0``, then a range
        test) because evoxc raises on out-of-range gathers; the gather's mask then
        overwrites every miss with ``I_INF``, so the sanitized ``0`` never leaks and
        a block of sanitized keys is ``trusted``.
    """
    key_span = _positive_static(key_span, "key_span")
    width = int(table.shape[1])
    if width != key_span + 1:
        raise ConfigError(
            f"key table width {width} does not match key_span + 1 ({key_span + 1})"
        )
    return _slot_of(table, keys, key_span)


def _slot_of(table, keys, key_span):
    """Trusted slot lookup: sanitized indices, gather, ``I_INF`` on a miss.

    Private body of :func:`slot_lookup`, without its width check.
    """
    indices, in_range = _key_indices(keys, key_span)
    flat_index = prepare_flat_index(table, indices, in_range, 0, trusted=True)
    values = flat_gather_prepared(table, flat_index, indices.shape)
    return evoxc.select(in_range, values, _scalar_constant(I_INF, evoxc.int32))


def edge_order(dst, edge_valid, max_nodes, max_conns):
    """In-edge rank of every connection row plus per-node in-degree prefixes.

    Args:
        dst: ``(P, C)`` ``int32`` destination slots; ``I_INF`` where the row is
            absent or unmatched; edge_valid: ``(P, C)`` ``bool`` from the lookup.
        max_nodes (int): ``N``; max_conns (int): ``C``.

    Returns:
        tuple: ``(rank, cc)``. ``rank`` ``(P, C)`` ``int32`` is the number of valid
        rows before row ``c`` in ``(dst, connection index)`` order (``I_INF`` on an
        invalid row); ``cc`` ``(P, N)`` ``int32`` counts the valid rows whose
        destination slot is ``<= i``, so node ``i``'s in-edge group starts at
        ``cc[i - 1]`` (``0`` for ``i == 0``) and an edge's in-group column is
        ``rank - start[dst]``.

    Notes:
        **One** stable ``argsort`` over the ``C + N`` combined entries — edge rows
        keyed ``2 * dst``, node slots keyed ``2 * slot + 1`` — yields both answers
        from its inverse ``inv``: ``rank[c] = inv[c] - dst[c]`` (exactly ``dst[c]``
        node entries sort below an edge at slot ``dst[c]``, ``stable=True`` keeping
        connection order inside a group) and ``cc[i] = inv[C + i] - i``. Keys stay in
        ``[0, 2 * N + 2]`` (an edge key precedes its own node key, an invalid row is
        keyed past the last one), so :func:`_narrow_int` picks a radix-sortable
        carrier: ``int8`` for ``N <= 62``, ``int16`` for ``N <= 16382``, else ``int32``.
    """
    max_nodes = _positive_static(max_nodes, "max_nodes")
    max_conns = _positive_static(max_conns, "max_conns")
    pop_size, width = dst.shape
    pop_size, width = int(pop_size), int(width)
    if width != max_conns:
        raise ConfigError(f"dst must have {max_conns} columns, got {width}")
    key_dtype = _narrow_int(2 * max_nodes + 2)
    slots = _node_slots(pop_size, max_nodes)
    zero = enp.zeros((pop_size, max_conns), dtype=evoxc.int32)
    two = _scalar_constant(2, key_dtype)
    edge_key = evoxc.select(
        edge_valid,
        evoxc.multiply(evoxc.cast(evoxc.select(edge_valid, dst, zero), key_dtype), two),
        _scalar_constant(2 * max_nodes + 2, key_dtype),
    )
    node_key = evoxc.add(
        evoxc.multiply(evoxc.cast(slots, key_dtype), two),
        _scalar_constant(1, key_dtype),
    )
    order = evoxc.argsort(
        evoxc.concatenate([edge_key, node_key], axis=1), axis=-1, stable=True
    )
    total = max_conns + max_nodes
    positions = _node_slots(pop_size, total)
    # ``order`` is a permutation of ``0..total-1`` on every row, so every cell of
    # the inverse is written exactly once and the base can be plain zeros.
    inverse = flat_scatter(
        enp.zeros((pop_size, total), dtype=evoxc.int32),
        evoxc.cast(order, evoxc.int32),
        positions,
        trusted=True,
    )
    safe_dst = evoxc.select(edge_valid, dst, zero)
    rank = evoxc.select(
        edge_valid,
        evoxc.subtract(evoxc.slice(inverse, (0, 0), (pop_size, max_conns)), safe_dst),
        _iinf_like(dst),
    )
    cc = evoxc.subtract(
        evoxc.slice(inverse, (0, max_conns), (pop_size, max_nodes)), slots
    )
    return rank, cc


def edge_group_rank(dst, edge_valid, max_conns):
    """Rank each connection row within the valid rows ordered by destination.

    Args:
        dst: ``(P, C)`` ``int32`` destination slots; ``I_INF`` on an invalid row;
            edge_valid: ``(P, C)`` ``bool`` from the lookup.
        max_conns (int): ``C``.

    Returns:
        ``(P, C)`` ``int32``: the position of row ``c`` among the valid rows ordered
        by ``(dst, connection index)``; ``I_INF`` for invalid rows.

    Notes:
        One ``argsort(..., stable=True)`` over the ``int32`` composite ``dst * C +
        conn_index`` (invalid rows sanitized to slot ``0`` first), then one
        ``flat_scatter`` of ``arange(C)`` to invert the permutation.
        :func:`derive_schedule` uses :func:`edge_order`, whose single sort yields
        this ``rank`` *and* ``cc`` on a radix-friendly key.
    """
    max_conns = _positive_static(max_conns, "max_conns")
    pop_size, width = dst.shape
    pop_size, width = int(pop_size), int(width)
    if width != max_conns:
        raise ConfigError(f"dst must have {max_conns} columns, got {width}")
    columns = _node_slots(pop_size, max_conns)
    zero = enp.zeros((pop_size, max_conns), dtype=evoxc.int32)
    safe_dst = evoxc.select(edge_valid, dst, zero)
    composite = evoxc.add(
        evoxc.multiply(safe_dst, _scalar_constant(max_conns, evoxc.int32)), columns
    )
    order = evoxc.argsort(
        evoxc.select(edge_valid, composite, _iinf_like(composite)), axis=-1, stable=True
    )
    inverse = flat_scatter(
        enp.zeros((pop_size, max_conns), dtype=evoxc.int32),
        evoxc.cast(order, evoxc.int32),
        columns,
        trusted=True,
    )
    return evoxc.select(edge_valid, inverse, _iinf_like(inverse))


def cum_in_degree(dst, edge_valid, max_nodes):
    """Count, per node, how many valid edges point at it or at a lower slot.

    Args:
        dst: ``(P, C)`` ``int32`` destination slots (``I_INF`` for invalid rows);
            edge_valid: ``(P, C)`` ``bool``; max_nodes (int): ``N``.

    Returns:
        ``(P, N)`` ``int32``: ``cc[p, i]`` = number of valid edges of genome ``p``
        whose destination slot is ``<= i`` — the ``cumsum``-free ``cc`` of
        :func:`edge_order`, kept separately so the step stays testable alone.
    """
    max_nodes = _positive_static(max_nodes, "max_nodes")
    _rank, cc = edge_order(dst, edge_valid, max_nodes, int(dst.shape[1]))
    return cc


def incoming_table(rank, cc, dst, src, edge_valid, max_nodes, max_in_degree):
    """Pack each node's incoming edges into the static ``(P, N, K)`` table.

    Args:
        rank: ``(P, C)`` ``int32`` in-edge rank from :func:`edge_order``; cc:
            ``(P, N)`` ``int32`` cumulative in-degree from the same call.
        dst: ``(P, C)`` ``int32`` destination slots; src: the source slots;
            edge_valid: ``(P, C)`` ``bool``.
        max_nodes (int): ``N``; max_in_degree (int): ``K``.

    Returns:
        tuple: ``(in_conn, in_src, edge_overflow)`` — ``(P, N, K)`` ``int32``
        connection row indices, ``(P, N, K)`` ``int32`` source slots (``I_INF`` in
        the same positions) and ``(P,)`` ``int32`` counts of valid edges whose
        column reached ``K``.

    Notes:
        ``start[i] = cc[i - 1]`` (``0`` for ``i == 0``) and ``col = rank -
        start[dst]``; the flat scatter target ``dst * K + col`` lands in a ``(P, N *
        K + 1)`` buffer whose trailing cell is the dump slot, so edges with ``col
        >= K`` or an invalid row go there instead of raising. ``edge_overflow`` is
        a ``reduce_sum`` of ``(col >= K) & edge_valid``: data, not an exception.
    """
    max_nodes = _positive_static(max_nodes, "max_nodes")
    max_in_degree = _positive_static(max_in_degree, "max_in_degree")
    pop_size, max_conns = dst.shape
    pop_size, max_conns = int(pop_size), int(max_conns)
    zero = enp.zeros((pop_size, max_conns), dtype=evoxc.int32)
    if max_nodes > 1:
        start = evoxc.concatenate(
            [
                enp.zeros((pop_size, 1), dtype=evoxc.int32),
                evoxc.slice(cc, (0, 0), (pop_size, max_nodes - 1)),
            ],
            axis=1,
        )
    else:
        start = enp.zeros((pop_size, 1), dtype=evoxc.int32)
    safe_dst = evoxc.select(edge_valid, dst, zero)
    column = evoxc.subtract(
        rank,
        flat_gather_prepared(
            start, prepare_flat_index(start, safe_dst, trusted=True), safe_dst.shape
        ),
    )
    kept_column = evoxc.select(edge_valid, column, zero)
    in_window = enp.logical_and(
        enp.greater_equal(kept_column, 0), enp.less(kept_column, max_in_degree)
    )
    keep = enp.logical_and(edge_valid, in_window)
    target = evoxc.add(
        evoxc.multiply(safe_dst, _scalar_constant(max_in_degree, evoxc.int32)),
        evoxc.select(keep, kept_column, zero),
    )
    block = _window_block(target, keep, pop_size, max_nodes, max_in_degree)
    in_conn = _window_table(
        block, _node_slots(pop_size, max_conns), pop_size, max_nodes, max_in_degree
    )
    in_src = _window_table(block, src, pop_size, max_nodes, max_in_degree)
    edge_overflow = evoxc.cast(
        evoxc.reduce_sum(
            evoxc.cast(
                enp.logical_and(edge_valid, enp.greater_equal(kept_column, max_in_degree)),
                evoxc.int32,
            ),
            axes=(1,),
        ),
        evoxc.int32,
    )
    return in_conn, in_src, edge_overflow


def _window_block(target, keep, pop_size, max_nodes, max_in_degree):
    """Flat scatter block for the ``(P, N*K + 1)`` window buffer, built once.

    ``target = dst * K + col`` is provably inside ``[0, N*K)``; a row without
    ``keep`` is routed to the dump cell ``N*K``, so the block is ``trusted``.
    """
    width = max_nodes * max_in_degree
    return prepare_flat_index((pop_size, width + 1), target, keep, width, trusted=True)


def _window_table(block, updates, pop_size, max_nodes, max_in_degree):
    """Scatter one payload into a fresh ``(P, N, K)`` window table.

    The shared body of :func:`incoming_table`'s two passes: an ``(P, N*K + 1)``
    ``I_INF`` buffer written through ``block`` (dropped rows land in its trailing
    cell), the dump column sliced off and the rest reshaped to the window.
    """
    width = max_nodes * max_in_degree
    buffer = flat_scatter_prepared(
        enp.full((pop_size, width + 1), I_INF, dtype=evoxc.int32), block, updates
    )
    return evoxc.reshape(
        evoxc.slice(buffer, (0, 0), (pop_size, width)),
        (pop_size, max_nodes, max_in_degree),
    )


def node_levels(in_conn, in_src, input_slot, max_nodes, acyclic=False, edges=None):
    """Longest-path depth of every node (the level-synchronous schedule).

    Args:
        in_conn: ``(P, N, K)`` ``int32`` incoming-edge table (``I_INF`` padding);
            only its shape is read here. in_src: the ``(P, N, K)`` ``int32`` source
            slots of those edges, ``I_INF`` exactly where padded.
        input_slot: ``(P, I)`` ``int32`` slots of the input nodes, pinned to 0;
            max_nodes (int): ``N``, the slot count and the scale of the loop cap.
        acyclic (bool): ``False`` (default) keeps the live-cycle semantics, paid
            only where a cycle exists: the level-only relaxation runs first,
            :func:`_cycle_violation` tests its output, and the cycle-detecting
            fallback runs inside ``evoxc.cond`` only for a failing population.
            ``True`` asserts a cycle-free population and drops test and fallback, so
            a cyclic genome receives a saturated (wrong) level instead of ``I_INF``.
        edges: ``(src, dst, edge_valid)`` — the ``(P, C)`` endpoint slots and the
            "both endpoints matched" mask of the connection rows; required when
            ``acyclic`` is ``False`` as the fallback's trigger.

    Returns:
        ``(P, N)`` ``int32`` levels: ``0`` for input nodes, ``max(pred) + 1``
        otherwise, ``I_INF`` for an absent node, a node unreachable from the inputs,
        or (only with ``acyclic=False``) a node on / downstream of a cycle; raises
        ``ConfigError`` for a bad static or a missing ``edges`` triple.

    Notes:
        A max-plus relaxation ``x[i] = max(x[i], max_k(x[in_src[i, k]] + 1))`` in an
        ``evoxc.while_loop``, with ``-1`` = "not settled" and ``-2`` the dump cell's
        value (so an unsettled predecessor is neutral and no mask is needed). The
        window's flat index block depends only on ``in_src`` and is built once
        outside the loop; an iteration concatenates its ``(P, N)`` state to a
        one-cell ``-2`` dump column, gathers through that block and folds the ``K``
        candidates pairwise.

        ``level`` is seeded at ``0`` on the input slots, re-pinned every iteration,
        and at ``-1`` elsewhere. The fallback also relaxes ``reach``, the longest
        *walk* ending at each node, seeded at ``0`` everywhere and never re-pinned,
        so it grows only where a cycle is reachable (a walk may start on the cycle,
        which is why a pinned input does not freeze it); snapshotting it after
        iteration ``N`` of a loop that may run ``2N`` separates those nodes from the
        acyclic ones, and a ``level`` still ``-1`` marks a node unreachable from the
        inputs. Both campaigns are one ``settle(cap, detect_cycles)``.
    """
    max_nodes = _positive_static(max_nodes, "max_nodes")
    pop_size, nodes, width = in_conn.shape
    pop_size, nodes, width = int(pop_size), int(nodes), int(width)
    if nodes != max_nodes:
        raise ConfigError(f"in_conn node axis {nodes} does not match max_nodes {max_nodes}")
    edge_src = edge_dst = edge_valid = None
    if not acyclic:
        if edges is None:
            raise ConfigError(
                "node_levels: acyclic=False needs the (src, dst, edge_valid) triple "
                "of the connection rows for its in-graph cycle test"
            )
        edge_src, edge_dst, edge_valid = edges
        for name, tensor in (
            ("src", edge_src),
            ("dst", edge_dst),
            ("edge_valid", edge_valid),
        ):
            if int(tensor.shape[0]) != pop_size:
                raise ConfigError(
                    f"node_levels: {name} population axis {int(tensor.shape[0])} does "
                    f"not match in_conn {pop_size}"
                )
            if int(tensor.shape[1]) != int(edge_src.shape[1]):
                raise ConfigError(
                    f"node_levels: {name} connection axis {int(tensor.shape[1])} does "
                    f"not match src {int(edge_src.shape[1])}"
                )
    # The dump column sits one past the last node slot and holds ``-2``: after the
    # ``+1`` of the relaxation that is ``-1``, i.e. "this edge contributes nothing",
    # so neither the gathered block nor the reduction needs a mask. ``in_src`` is
    # ``I_INF`` in exactly the padded window positions, so clamping it to the slot
    # count routes those at the dump column without a separate mask and without an
    # ``(P, N, K)`` constant.
    index = evoxc.minimum(in_src, _scalar_constant(nodes, evoxc.int32))
    flat_index = prepare_flat_index((pop_size, nodes + 1), index)
    pinned = _pinned_mask(input_slot, _node_slots(pop_size, nodes))
    step_one = _scalar_constant(1, evoxc.int32)

    def settle(cap, detect_cycles):
        """One relaxation campaign; ``(level, blank)`` in the raw carrier dtype."""
        value_dtype, no_snapshot = _carrier(cap)
        dump = enp.full((pop_size, 1), -2, dtype=value_dtype)
        zero_level = enp.zeros((pop_size, nodes), dtype=value_dtype)
        minus_one = enp.full((pop_size, nodes), -1, dtype=value_dtype)
        one = _scalar_constant(1, value_dtype)

        def gather_best(values):
            """Per-node max over the gathered incoming sources (``-2`` = padding)."""
            source = flat_gather_prepared(
                evoxc.concatenate([values, dump], axis=1),
                flat_index,
                (pop_size, nodes, width),
            )
            best = evoxc.slice(source, (0, 0, 0), (pop_size, nodes, 1))
            for k in range(1, width):
                best = evoxc.maximum(
                    best, evoxc.slice(source, (0, 0, k), (pop_size, nodes, 1))
                )
            return evoxc.reshape(best, (pop_size, nodes))

        def relax(values):
            """One max-plus step of a longest-walk relaxation (``-1`` = unsettled)."""
            best = gather_best(values)
            # ``best < 0`` means "no settled predecessor": either every slot is the
            # ``-2`` dump or the best real predecessor is itself still ``-1``. Both
            # must leave ``values`` untouched instead of settling the node at ``0``
            # (``maximum(values, -1)`` is ``values`` because ``values >= -1``).
            return evoxc.maximum(
                values,
                evoxc.select(enp.greater(best, -1), evoxc.add(best, one), minus_one),
            )

        def keep_going(state):
            """Loop condition; both campaigns carry ``(step, ..., changed)``."""
            return enp.logical_and(enp.less(state[0], cap), state[-1])

        if not detect_cycles:
            def body(state):
                step, level, _changed = state
                next_level = evoxc.select(pinned, zero_level, relax(level))
                next_step = evoxc.add(step, step_one)
                moved = enp.not_equal(next_level, level)
                return (next_step, next_level, evoxc.reduce_max(moved))

            _step, level, _changed = evoxc.while_loop(
                keep_going,
                body,
                (
                    enp.zeros((), dtype=evoxc.int32),
                    evoxc.select(pinned, zero_level, minus_one),
                    enp.full((), True, dtype=evoxc.bool_),
                ),
            )
            blank = enp.logical_and(enp.logical_not(pinned), enp.less(level, 0))
        else:
            def body(state):
                step, level, reach, reach_at, _changed = state
                next_level = evoxc.select(pinned, zero_level, relax(level))
                next_reach = relax(reach)
                next_step = evoxc.add(step, step_one)
                snapshot = enp.equal(next_step, max_nodes)
                moved = enp.logical_or(
                    enp.not_equal(next_level, level), enp.not_equal(next_reach, reach)
                )
                return (
                    next_step,
                    next_level,
                    next_reach,
                    evoxc.select(snapshot, next_reach, reach_at),
                    evoxc.reduce_max(moved),
                )

            _step, level, reach, reach_at, _changed = evoxc.while_loop(
                keep_going,
                body,
                (
                    enp.zeros((), dtype=evoxc.int32),
                    evoxc.select(pinned, zero_level, minus_one),
                    zero_level,
                    enp.full((pop_size, nodes), no_snapshot, dtype=value_dtype),
                    enp.full((), True, dtype=evoxc.bool_),
                ),
            )
            # growth between the snapshots, or never reached from the inputs
            blank = enp.logical_and(
                enp.logical_not(pinned),
                enp.logical_or(enp.greater(reach, reach_at), enp.less(level, 0)),
            )
        return level, blank

    def answer(level, blank):
        """Map a campaign's raw carrier levels onto the public ``int32`` answer."""
        level = evoxc.cast(level, evoxc.int32)
        return evoxc.select(blank, _iinf_like(level), level)

    fast_level, fast_blank = settle(max_nodes, False)
    if acyclic:
        return answer(fast_level, fast_blank)
    return evoxc.cond(
        _cycle_violation(fast_level, edge_src, edge_dst, edge_valid, nodes),
        lambda: answer(*settle(2 * max_nodes, True)),
        lambda: answer(fast_level, fast_blank),
    )


def _carrier(cap):
    """``(dtype, no_snapshot)``: the narrowest integer carrier for a ``cap``-step loop.

    A value lies in ``[-2, cap]``, so ``int8`` serves ``cap <= 127`` and ``int16``
    serves ``cap < 32767`` — a static choice, never a data-dependent one.
    ``no_snapshot`` is the cycle-detecting campaign's "no snapshot yet" sentinel,
    above every reachable walk length (``I_INF`` for ``int32``).
    """
    if cap <= 127:
        return evoxc.int8, 127
    if cap < 32767:
        return evoxc.int16, 32767
    return evoxc.int32, I_INF


def _cycle_violation(level, src, dst, edge_valid, max_nodes):
    """0-d bool: some member has an edge whose settled levels are not increasing.

    Args:
        level: ``(P, N)`` levels of the level-only relaxation **before** the
            ``I_INF`` mapping — ``-1`` for "never settled", ``0 .. N`` elsewhere.
        src: ``(P, C)`` ``int32`` source slots (``I_INF`` on an absent or unmatched
            row); dst: the destination slots, likewise; edge_valid: ``(P, C)``
            ``bool`` = a real row whose endpoint keys both matched.
        max_nodes (int): ``N``.

    Returns:
        A 0-d ``bool``: ``True`` iff at least one member has a valid edge ``(u, v)``
        with ``level[u] >= level[v]``.

    Notes:
        The exactness test of the ``acyclic=False`` fast path: settled levels obey
        ``level[v] = max(level[preds]) + 1 > level[u]`` on every edge, and every
        cycle breaks it — a reachable one grows until the cap, one through a pinned
        input leaves its non-pinned successors growing, an unreachable one leaves
        both endpoints at ``-1``. An *unreachable edge* violates it too, which costs
        a fallback and never a wrong answer. Two ``(P, C)`` gathers over the ``(P, N
        + 1)`` vector, each slot clamped to the trailing ``-2`` dump cell (so no
        index is out of range) and ``edge_valid`` masking the half-matched rows;
        then two ``reduce_max`` steps — data for ``evoxc.cond``.
    """
    pop_size, nodes = int(level.shape[0]), int(level.shape[1])
    padded = evoxc.concatenate(
        [level, enp.full((pop_size, 1), -2, dtype=level.dtype)], axis=1
    )

    def levels_at(slots):
        """``(P, C)`` levels of those slots; ``I_INF`` reads the ``-2`` dump cell."""
        index = evoxc.minimum(slots, _scalar_constant(nodes, evoxc.int32))
        return flat_gather_prepared(
            padded,
            prepare_flat_index((pop_size, nodes + 1), index),
            (int(slots.shape[0]), int(slots.shape[1])),
        )

    backward = enp.logical_and(
        edge_valid, enp.greater_equal(levels_at(src), levels_at(dst))
    )
    per_member = evoxc.reduce_max(evoxc.cast(backward, evoxc.int32), axes=(1,))
    return enp.greater(evoxc.reduce_max(per_member), _scalar_constant(0, evoxc.int32))


def _pinned_mask(input_slot, slots):
    """``(P, N)`` bool mask of the node slots named by ``input_slot``.

    A broadcast equality over the ``I`` axis (``O(N * I)``) instead of a scatter;
    ``I_INF`` never matches a real slot, so no sanitizing or dump column.
    """
    pop_size, num_inputs = input_slot.shape
    pop_size, num_inputs = int(pop_size), int(num_inputs)
    nodes = int(slots.shape[1])
    targets = enp.broadcast_to(
        evoxc.reshape(input_slot, (pop_size, 1, num_inputs)), (pop_size, nodes, num_inputs)
    )
    sources = enp.broadcast_to(
        evoxc.reshape(slots, (pop_size, nodes, 1)), (pop_size, nodes, num_inputs)
    )
    hits = evoxc.cast(enp.equal(targets, sources), evoxc.int32)
    return enp.greater(evoxc.reduce_max(hits, axes=(2,)), 0)


# ============================================================ host-side checker
def validate_schedule(transformed):
    """Check :class:`Transformed` invariants on the host (test/debug helper).

    Args:
        transformed (Transformed): A schedule from :func:`derive_schedule` whose
            arrays are already host memory (``evoxc.Tensor`` leaves are read through
            :func:`~tensorneat.common.tools.to_numpy`).

    Returns:
        None: Returns ``None`` when every invariant of :class:`Transformed` holds.

    Raises:
        tensorneat.common.errors.ConfigError: Describing the *first* violated
            invariant — a shape or dtype outside the documented signature,
            ``in_conn``/``in_src`` disagreeing on a padded position, ``in_valid``
            disagreeing with ``in_conn != I_INF``, ``topo_key``/``node_level``
            disagreeing, a negative ``edge_overflow``, or an ``input_slot`` /
            ``output_slot`` outside ``[0, N)``.
    """
    src = _host_int(transformed.src, "src", 2)
    dst = _host_int(transformed.dst, "dst", 2)
    edge_valid = _host_bool(transformed.edge_valid, "edge_valid", 2)
    in_conn = _host_int(transformed.in_conn, "in_conn", 3)
    in_src = _host_int(transformed.in_src, "in_src", 3)
    in_valid = _host_bool(transformed.in_valid, "in_valid", 3)
    node_level = _host_int(transformed.node_level, "node_level", 2)
    node_exists = _host_bool(transformed.node_exists, "node_exists", 2)
    topo_key = _host_int(transformed.topo_key, "topo_key", 2)
    input_slot = _host_int(transformed.input_slot, "input_slot", 2)
    output_slot = _host_int(transformed.output_slot, "output_slot", 2)
    edge_overflow = _host_int(transformed.edge_overflow, "edge_overflow", 1)

    pop_size, num_conns = src.shape
    num_nodes, width = in_conn.shape[1], in_conn.shape[2]
    for name, array, expected in (
        ("dst", dst, (pop_size, num_conns)),
        ("edge_valid", edge_valid, (pop_size, num_conns)),
        ("in_conn", in_conn, (pop_size, num_nodes, width)),
        ("in_src", in_src, (pop_size, num_nodes, width)),
        ("in_valid", in_valid, (pop_size, num_nodes, width)),
        ("node_level", node_level, (pop_size, num_nodes)),
        ("node_exists", node_exists, (pop_size, num_nodes)),
        ("topo_key", topo_key, (pop_size, num_nodes)),
        ("edge_overflow", edge_overflow, (pop_size,)),
    ):
        if array.shape != expected:
            raise ConfigError(
                f"validate_schedule: {name} has shape {array.shape}, expected {expected}"
            )
    for name, array in (("input_slot", input_slot), ("output_slot", output_slot)):
        if array.shape[0] != pop_size:
            raise ConfigError(
                f"validate_schedule: {name} has population axis {array.shape[0]}, "
                f"expected {pop_size}"
            )

    padded = in_conn == I_INF
    if not np.array_equal(padded, in_src == I_INF):
        raise ConfigError(
            "validate_schedule: in_conn and in_src must be I_INF in exactly the same "
            "positions (a padded window slot disagrees)"
        )
    if not np.array_equal(in_valid, ~padded):
        raise ConfigError(
            "validate_schedule: in_valid must be exactly in_conn != I_INF "
            "(a padded window slot disagrees)"
        )
    dead_level = node_level == I_INF
    if not np.all(topo_key[dead_level] == I_INF):
        raise ConfigError(
            "validate_schedule: an I_INF node_level must pair with an I_INF topo_key"
        )
    live_level = ~dead_level
    if np.any(topo_key[live_level] == I_INF):
        raise ConfigError(
            "validate_schedule: a finite node_level must pair with a finite topo_key "
            "(topo_key is I_INF on a live node)"
        )
    expected_topo = node_level.astype(np.int64) * num_nodes + np.arange(
        num_nodes, dtype=np.int64
    )[None, :]
    mismatch = live_level & (topo_key != expected_topo)
    if np.any(mismatch):
        member, slot = (int(index) for index in np.argwhere(mismatch)[0])
        raise ConfigError(
            "validate_schedule: topo_key must equal node_level * N + slot on every "
            f"live node (N = {num_nodes}); at (member {member}, slot {slot}) "
            f"node_level is {int(node_level[member, slot])} and topo_key is "
            f"{int(topo_key[member, slot])}"
        )
    if np.any(edge_overflow < 0):
        raise ConfigError(
            "validate_schedule: edge_overflow is a count of dropped edges and cannot be "
            "negative"
        )
    for name, array in (("input_slot", input_slot), ("output_slot", output_slot)):
        live = array != I_INF
        if np.any(array[live] < 0) or np.any(array[live] >= num_nodes):
            raise ConfigError(
                f"validate_schedule: {name} must hold slots in [0, {num_nodes}) or "
                f"I_INF, got {array[live]}"
            )
    return None


def _host_array(value, name):
    """Host-side numpy view of a ``transformed`` field (validation path only).

    Converts through :func:`~tensorneat.common.tools.to_numpy` and rejects a
    non-array field.
    """
    array = np.asarray(to_numpy(value))
    if array.dtype == object:
        raise ConfigError(f"validate_schedule: {name} is not a host array")
    return array


def _host_int(value, name, ndim):
    """Host integer field of rank ``ndim`` (``int32``-valued)."""
    array = _host_array(value, name)
    if array.dtype.kind not in "iu":
        raise ConfigError(
            f"validate_schedule: {name} must be an integer array, got dtype {array.dtype}"
        )
    if array.ndim != ndim:
        raise ConfigError(
            f"validate_schedule: {name} must have rank {ndim}, got shape {array.shape}"
        )
    return array


def _host_bool(value, name, ndim):
    """Host bool field of rank ``ndim``."""
    array = _host_array(value, name)
    if array.dtype != np.bool_:
        raise ConfigError(
            f"validate_schedule: {name} must be a bool array, got dtype {array.dtype}"
        )
    if array.ndim != ndim:
        raise ConfigError(
            f"validate_schedule: {name} must have rank {ndim}, got shape {array.shape}"
        )
    return array
