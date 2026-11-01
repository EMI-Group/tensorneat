"""Tensorization: turn a ragged evolved topology into a static execution schedule.

Dependency DAG position: ``tensorneat.common`` bottom layer. Imports
:mod:`~tensorneat.common.tools` (flat-index helpers), :mod:`~tensorneat.common.errors`
and ``evoxc``; imports nothing from ``genome``, ``algorithm``, ``problem`` or
``pipeline``, and no ``jax``/``torch``. This module replaces the pre-rewrite
``common/graph.py``: the JAX-era topological sort over a dense ``(N, N)``
adjacency matrix is gone, together with its ``max_nodes ** 2`` cost.

What a schedule is
------------------
NEAT topologies are ragged — different population members have different node
and connection counts — but evoxc kernels need static shapes. Tensorization
answers *how to execute* one whole population with fixed-shape arrays:

* every connection row is resolved to ``(src, dst)`` **node slots** (slot
  indices into that genome's own arrays, never innovation numbers),
* the incoming edges of each node are packed into a ``(P, N, K)`` table, so a
  forward pass is a level-ordered sweep whose work is proportional to the real
  connections and the network depth (never to ``max_nodes ** 2``),
* each node gets a ``node_level`` (longest-path depth), which is a valid
  level-sync topological order for feedforward execution.

Output contract: :class:`Transformed` holds **arrays only** — no Python
statics, no callables — so it is a plain pytree that can cross a graph
boundary and be cached per shape signature. Any static fact a consumer needs
(``max_conns``, ``max_in_degree``, ``key_span``, ``pop_size``) is implied by a
shape or a dtype.

Cost: ``O(P * (S + C log C + N * K * D))`` with ``S = key_span``,
``C = max_conns``, ``K = max_in_degree`` and ``D`` = network depth (the level
relaxation's trip count; ``2N`` only for a population that contains a live
cycle) — i.e. edge-proportional apart from the padded ``N * K`` table and the
one-off key→slot table. ``derive_schedule`` runs once per generation (inside
``Pipeline.step``), so that cost is amortized over the whole generation.

Deliberate evoxc constraints (validated; see ``tools`` for the full list):
no ``cumsum`` (a StableHLO export deferral, hence the ``cc`` counting trick),
no scatter-with-reduce, no batched ``take_along_axis`` (hence flat indexing),
no ``where`` (``evoxc.select``), and out-of-range indices raise instead of
clamping — every index into a table must therefore be sanitized *and* masked.

Everything here is in-graph: no host round-trip, no ``.numpy()``, and no
Python ``if`` on a traced value (only on statics such as dtypes, shapes and the
declared static integers, which are baked into the trace).
"""

from typing import Any, NamedTuple

import evoxc
import evoxc.numpy as enp
import numpy as np

from .errors import ConfigError
from .tools import (
    I_INF,
    KEY_MAX,
    flat_gather,
    flat_scatter,
    masked_flat_gather,
    masked_flat_scatter,
)


class Transformed(NamedTuple):
    """Static execution schedule of one population (all fields batch-first).

    Every field has population size ``P`` on axis 0; ``N = max_nodes``,
    ``C = max_conns``, ``K = max_in_degree``, ``I = len(input_keys)`` and
    ``O = len(output_keys)`` are static at build time. Sentinels: ``I_INF``
    (``2147483647``, ``int32`` max) means "no index / unmatched / unavailable"
    in the integer fields; ``NaN`` never appears here (these are integer/bool
    arrays), it lives in the gene payload instead.

    The fields are exactly the twelve below, in this order; ``level`` and
    ``overflow`` are read-only compatibility properties (see the end of the
    class), never fields.

    Invariances consumers may rely on:

    * ``in_conn``, ``in_src`` and ``in_valid`` mark the same positions
      (``in_valid == (in_conn != I_INF)``), and a node's columns ``0..d-1`` are
      dense whenever its in-degree is ``d``; columns ``d..K-1`` are padding.
    * for every ``(p, c)`` with ``edge_valid[p, c]``, exactly one
      ``(p, i, k)`` satisfies ``in_conn[p, i, k] == c`` and
      ``in_src[p, i, k] == src[p, c]`` — except edges whose in-group rank
      reached ``K``, which are counted in ``edge_overflow``.
    * ``in_conn[p, i, 0] == I_INF`` means "node ``i`` has no incoming edge"
      (an input node, or an isolated node), i.e. ``in_valid[p, i, 0]`` is
      ``False``.
    * ``node_level[p, i]`` is ``I_INF`` iff node ``i`` is absent
      (``~node_exists[p, i]``), unreachable from the inputs, or on / downstream
      of a live cycle; ``reduce_max(node_level)`` bounds the execution loop.
    * executing levels ``0, 1, ...`` in order is a valid topological order;
      nodes within one level are independent.
    * ``edge_overflow[p] > 0`` means the schedule is lossy for genome ``p``:
      the host must rebuild with a larger ``max_in_degree`` (or raise
      :class:`~tensorneat.common.errors.ShapeBoundError`). Kernels never
      silently drop edges.
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
def _positive_int(value: Any, name: str) -> int:
    """Return ``value`` as a positive Python ``int`` (a static, validated at trace time)."""
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ConfigError(f"{name} must be an int, got {type(value).__name__}")
    value = int(value)
    if value <= 0:
        raise ConfigError(f"{name} must be positive, got {value}")
    return value


def _check_shape(tensor: Any, expected: tuple, name: str) -> None:
    """Raise :class:`ConfigError` unless ``tensor``'s static shape is ``expected``."""
    shape = tuple(int(size) for size in tensor.shape)
    if shape != tuple(expected):
        raise ConfigError(f"{name} must have shape {tuple(expected)}, got {shape}")


def _scalar_int(value: int) -> Any:
    """A rank-0 ``int32`` tensor — the dtype-safe operand for in-graph arithmetic."""
    return evoxc.cast(value, evoxc.int32)


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

    Accepts both sentinel spellings: ``int32`` keys use ``I_INF`` (exact integer
    comparison) and ``float32`` keys use ``NaN``. This is a *trace-time* branch
    on the static dtype, never a runtime one.

    Args:
        keys: ``int32`` or ``float32`` key array of any shape.

    Returns:
        Bool tensor of ``keys``' shape, ``True`` where the entry is a real key.

    Notes:
        The float branch is "not ``NaN`` and below ``KEY_MAX``" — evoxc has no
        ``isfinite``; a negative key is filtered out later by the ``[0, S)``
        range test in :func:`_key_indices`, so the two spellings agree on every
        key that could ever be a real node key.
    """
    if keys.dtype.kind in ("i", "u"):
        return enp.not_equal(keys, _iinf_like(keys))
    return enp.logical_and(
        enp.logical_not(evoxc.isnan(keys)),
        enp.less(keys, enp.full(tuple(keys.shape), KEY_MAX, dtype=keys.dtype)),
    )


def _key_indices(keys: Any, key_span: int) -> tuple:
    """Sanitized table indices of a key array plus the "indexable" mask.

    Args:
        keys: ``int32``/``float32`` key array of any shape.
        key_span (int): ``S``, the exclusive key bound of the dense table.

    Returns:
        tuple: ``(idx, in_range)`` where ``idx`` is ``int32`` of ``keys``' shape
        with every non-indexable entry replaced by ``0`` (never ``NaN``, never
        ``I_INF``) and ``in_range`` is the bool mask of the entries whose key is
        a real node key inside ``[0, key_span)``.

    Notes:
        evoxc's numpy backend **raises** on an out-of-range gather/scatter, so
        the index must be safe *before* it is used and the result must be masked
        afterwards. A present key that does not fit ``key_span`` cannot be
        refused here — the values are traced, not host data — so it is treated
        as a miss (`in_range == False`), exactly like an unknown key.
    """
    present = _key_present(keys)
    zero = enp.zeros(tuple(keys.shape), dtype=keys.dtype)
    in_range = enp.logical_and(
        present,
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
):
    """Derive the execution schedule of one whole population (one evoxc graph).

    Args:
        node_keys: ``(P, max_nodes)`` ``int32`` (``I_INF`` padding) or
            ``float32`` (``NaN`` padding) — the node **key column**
            (``nodes[:, :, 0]``). Both sentinel spellings are accepted; the
            dispatch is on the static dtype (see :func:`_key_present`).
        conn_in: ``(P, max_conns)`` endpoint key of each connection row's
            source, same dtype and padding as ``node_keys``.
        conn_out: ``(P, max_conns)`` destination endpoint key, likewise.
        input_keys: ``(I,)`` keys of the input nodes, shared by every genome in
            the batch (no batch axis), same dtype as ``node_keys``.
        output_keys: ``(O,)`` keys of the output nodes, likewise shared.
        pop_size (int): ``P``; trace-time static (part of the shape signature).
        max_nodes (int): ``N``, the padded node count; trace-time static.
        max_conns (int): ``C``, the padded connection-row count; trace-time
            static.
        max_in_degree (int): ``K``, the static in-degree bound of the
            ``(P, N, K)`` incoming table; trace-time static.
        key_span (int): ``S``, the static key-magnitude bound of the dense
            key→slot table (``(P, S + 1)``); trace-time static.

    The five statics are positional-or-keyword (not keyword-only), because
    ``evoxc.build`` mirrors a *positional* call while
    ``functools.partial(derive_schedule.fn, **statics)`` must keep working.

    Returns:
        Transformed: the schedule described by :class:`Transformed`. Arrays
        only — no Python statics — so the pytree can cross a graph boundary
        and be cached per shape signature.

    Raises:
        tensorneat.common.errors.ConfigError: At trace time, for a non-positive
            or shape-inconsistent static, or for key arrays of differing
            dtypes/textures.
        evoxc.ShapeError / evoxc.BackendError: Propagated unchanged.

    Notes:
        Composition (each step is a separate, individually testable helper in
        this module):

        1. :func:`key_slot_table` — dense key→slot table ``(P, S+1) int32``,
           one flat scatter over the node keys, trailing dump column.
        2. :func:`slot_lookup` — ``src``/``dst`` from that table, with
           sanitized keys (sentinel → dump, clamp to ``[0, S-1]``); an
           unmatched endpoint is ``I_INF`` and makes the row invalid.
        3. :func:`edge_group_rank` — global rank of each edge in the
           ``(dst, conn_index)`` order.
        4. :func:`cum_in_degree` — ``cc[i]`` = number of edges with
           ``dst <= i``, via one stable ``argsort`` of the combined edge/node
           array and one flat scatter for the inverse permutation (this is the
           ``cumsum``-free counting trick).
        5. :func:`incoming_table` — ``in_conn``/``in_src``/``edge_overflow``
           from ``col = rank - start[dst]``; ``in_valid`` is then the free
           ``in_conn != I_INF`` mask.
        6. :func:`node_levels` — longest-path relaxation in a ``while_loop``,
           inputs pinned to level 0, cycles reported as ``I_INF``.
        7. ``node_exists``, ``topo_key``, ``input_slot`` and ``output_slot``
           are then free (``key != I_INF``/not ``NaN``, ``level * N + slot``,
           and two lookups of the shared key vectors).

        Value validation is impossible inside a trace: a key outside
        ``[0, key_span)`` and an input/output key that no node holds both
        surface as ``I_INF`` slots (a miss) instead of raising. Host code that
        owns the payload (the Pipeline, or a test) is where such a payload is
        refused — this kernel reports, it never guesses.
    """
    pop_size = _positive_int(pop_size, "pop_size")
    max_nodes = _positive_int(max_nodes, "max_nodes")
    max_conns = _positive_int(max_conns, "max_conns")
    max_in_degree = _positive_int(max_in_degree, "max_in_degree")
    key_span = _positive_int(key_span, "key_span")
    _check_shape(node_keys, (pop_size, max_nodes), "node_keys")
    _check_shape(conn_in, (pop_size, max_conns), "conn_in")
    _check_shape(conn_out, (pop_size, max_conns), "conn_out")
    if int(input_keys.shape[0]) > max_nodes:
        raise ConfigError(
            f"num_inputs {int(input_keys.shape[0])} exceeds max_nodes {max_nodes}"
        )
    if int(output_keys.shape[0]) > max_nodes:
        raise ConfigError(
            f"num_outputs {int(output_keys.shape[0])} exceeds max_nodes {max_nodes}"
        )
    for name, tensor in (("conn_in", conn_in), ("conn_out", conn_out)):
        if tensor.dtype != node_keys.dtype:
            raise ConfigError(
                f"{name} dtype {tensor.dtype} must match node_keys dtype {node_keys.dtype}"
            )
    for name, tensor in (("input_keys", input_keys), ("output_keys", output_keys)):
        if tensor.dtype != node_keys.dtype:
            raise ConfigError(
                f"{name} dtype {tensor.dtype} must match node_keys dtype {node_keys.dtype}"
            )

    table = key_slot_table(node_keys, key_span)
    src = slot_lookup(table, conn_in, key_span)
    dst = slot_lookup(table, conn_out, key_span)
    edge_valid = enp.logical_and(
        enp.logical_and(_key_present(conn_in), _key_present(conn_out)),
        enp.logical_and(
            enp.not_equal(src, _iinf_like(src)), enp.not_equal(dst, _iinf_like(dst))
        ),
    )
    input_slot = slot_lookup(table, _row_broadcast(input_keys, pop_size), key_span)
    output_slot = slot_lookup(table, _row_broadcast(output_keys, pop_size), key_span)
    rank = edge_group_rank(dst, edge_valid, max_conns)
    cc = cum_in_degree(dst, edge_valid, max_nodes)
    in_conn, in_src, edge_overflow = incoming_table(
        rank, cc, dst, src, edge_valid, max_nodes, max_in_degree
    )
    in_valid = enp.not_equal(in_conn, _iinf_like(in_conn))
    node_level = node_levels(in_conn, in_src, input_slot, max_nodes)
    node_exists = _key_present(node_keys)
    slots = _node_slots(pop_size, max_nodes)
    has_level = enp.not_equal(node_level, _iinf_like(node_level))
    ordered = evoxc.add(
        evoxc.multiply(evoxc.select(has_level, node_level, enp.zeros(tuple(node_level.shape), dtype=evoxc.int32)), _scalar_int(max_nodes)),
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
        node_keys: ``(P, N)`` ``int32`` (``I_INF`` padding) or ``float32``
            (``NaN`` padding) node key column, keys integer-valued in
            ``[0, key_span)``.
        key_span (int): ``S``, the static key-magnitude bound; trace-time
            static.

    Returns:
        ``(P, S + 1)`` ``int32`` table: column ``table[p, k]`` holds the slot
        of the node of genome ``p`` whose key is ``k``, or ``I_INF`` when no
        node holds that key. Column ``S`` is the trailing **dump slot**: rows
        whose key is absent (sentinel) are written there instead of index
        ``0``, which is what makes one unmasked ``flat_scatter`` safe (evoxc
        raises on out-of-range indices, and writing garbage into slot ``0``
        would corrupt a real node).

    Raises:
        tensorneat.common.errors.ConfigError: If ``key_span`` is not a
            positive ``int``, at trace time.
        evoxc.BackendError: Propagated unchanged.

    Notes:
        One :func:`~tensorneat.common.tools.masked_flat_scatter` of the slot
        indices, masked by "the key is real and inside ``[0, S)``". A key
        outside that range cannot be refused in-graph (values are traced), so
        it lands in the dump slot and later reads as a miss; under the float
        key spelling a key at or above ``2**24`` (``tools.KEY_MAX``) is also
        treated as absent, because it is no longer integer-exact in ``float32``.
    """
    key_span = _positive_int(key_span, "key_span")
    pop_size, max_nodes = node_keys.shape
    indices, in_range = _key_indices(node_keys, key_span)
    table = _iinf_like(enp.zeros((pop_size, key_span + 1), dtype=evoxc.int32))
    slots = _node_slots(int(pop_size), int(max_nodes))
    return masked_flat_scatter(table, indices, slots, in_range)


def slot_lookup(table, keys, key_span):
    """Look up node slots for a batch of keys, yielding ``I_INF`` on a miss.

    Args:
        table: ``(P, S + 1)`` ``int32`` output of :func:`key_slot_table`.
        keys: ``(P, L)`` key array to resolve (connection endpoints, or a
            broadcast shared input/output key vector), same dtype as the table's
            key column; ``NaN``/``I_INF`` allowed and always a miss.
        key_span (int): ``S``, matching the table; trace-time static.

    Returns:
        ``(P, L)`` ``int32`` slots in ``[0, N)``, with ``I_INF`` where the key
        is the padding sentinel, unknown, or outside ``[0, S)``.

    Raises:
        tensorneat.common.errors.ConfigError: If ``table``'s second dimension
            is not ``key_span + 1``.
        evoxc.ShapeError: Propagated unchanged.

    Notes:
        Indices are made safe *before* the gather (sentinel → ``0``, then a
        range test) because evoxc's numpy backend raises on out-of-range
        gathers; the sentinel/miss cases are then overwritten with ``I_INF`` by
        the gather's own mask, so the sanitized ``0`` never leaks into the
        result (``0`` is a real node slot).
    """
    key_span = _positive_int(key_span, "key_span")
    width = int(table.shape[1])
    if width != key_span + 1:
        raise ConfigError(
            f"key table width {width} does not match key_span + 1 ({key_span + 1})"
        )
    indices, in_range = _key_indices(keys, key_span)
    return masked_flat_gather(table, indices, mask=in_range, fill=I_INF)


def edge_group_rank(dst, edge_valid, max_conns):
    """Rank each connection row within its destination node's in-edge group.

    Args:
        dst: ``(P, C)`` ``int32`` destination slots; ``I_INF`` where the row is
            absent or unmatched.
        edge_valid: ``(P, C)`` ``bool`` from the lookup.
        max_conns (int): ``C``; trace-time static.

    Returns:
        ``(P, C)`` ``int32``: ``rank[p, c]`` = number of valid rows with the
        same ``dst`` and a smaller connection index, i.e. the position of row
        ``c`` in its destination's in-edge list. ``I_INF`` for invalid rows.

    Raises:
        tensorneat.common.errors.ConfigError: If ``max_conns`` is not positive
            or disagrees with ``dst``'s width.
        evoxc.ShapeError: Propagated unchanged.

    Notes:
        One ``argsort(..., stable=True)`` over the composite key
        ``dst * C + conn_index``, then one ``flat_scatter`` of ``arange(C)`` to
        invert the permutation. The composite is built in ``int64`` so that an
        invalid row can be keyed with ``I_INF`` ("after every valid row")
        without the ``I_INF * C`` overflow an ``int32`` composite would hit;
        valid keys are below ``N * C`` and therefore still sort first.
        ``stable=True`` keeps connection-row order inside a ``dst`` group,
        which is what makes the retained window deterministic.
    """
    max_conns = _positive_int(max_conns, "max_conns")
    pop_size, width = dst.shape
    pop_size, width = int(pop_size), int(width)
    if width != max_conns:
        raise ConfigError(f"dst must have {max_conns} columns, got {width}")
    columns = _node_slots(pop_size, max_conns)
    zero = enp.zeros((pop_size, max_conns), dtype=evoxc.int32)
    safe_dst = evoxc.select(edge_valid, dst, zero)
    composite = evoxc.add(
        evoxc.multiply(evoxc.cast(safe_dst, evoxc.int64), evoxc.cast(max_conns, evoxc.int64)),
        evoxc.cast(columns, evoxc.int64),
    )
    order = evoxc.argsort(
        evoxc.select(edge_valid, composite, _iinf_like(composite)), axis=-1, stable=True
    )
    inverse = flat_scatter(_iinf_like(safe_dst), evoxc.cast(order, evoxc.int32), columns)
    return evoxc.select(edge_valid, inverse, _iinf_like(inverse))


def cum_in_degree(dst, edge_valid, max_nodes):
    """Count, per node, how many valid edges point at it or at a lower slot.

    Args:
        dst: ``(P, C)`` ``int32`` destination slots (``I_INF`` for invalid
            rows); sanitized before use.
        edge_valid: ``(P, C)`` ``bool``.
        max_nodes (int): ``N``; trace-time static.

    Returns:
        ``(P, N)`` ``int32``: ``cc[p, i]`` = number of valid edges of genome
        ``p`` whose destination slot is ``<= i``. The start offset of node
        ``i``'s in-edge group is ``cc[i - 1]`` (``0`` for ``i == 0``), so the
        column of an edge is ``rank - start[dst]``.

    Raises:
        tensorneat.common.errors.ConfigError: If ``max_nodes`` is not positive.
        evoxc.ShapeError: Propagated unchanged.

    Notes:
        Deliberately ``cumsum``-free (``cumsum`` is a StableHLO export
        deferral in evoxc, numpy-interpreter only): build the combined array of
        ``C`` edge entries keyed ``2 * dst`` and ``N`` node entries keyed
        ``2 * slot + 1``, take one stable ``argsort``, invert it with one
        ``flat_scatter`` of ``arange(C + N)``, then read
        ``cc[i] = inverse[C + i] - i``. Edge keys precede their own node key
        (ties) so an edge is counted as "``<= i``"; invalid edges are keyed
        with ``I_INF`` and therefore sort after every node entry.
    """
    max_nodes = _positive_int(max_nodes, "max_nodes")
    pop_size, max_conns = dst.shape
    pop_size, max_conns = int(pop_size), int(max_conns)
    zero = enp.zeros((pop_size, max_conns), dtype=evoxc.int32)
    slots = _node_slots(pop_size, max_nodes)
    edge_key = evoxc.select(
        edge_valid,
        evoxc.multiply(
            evoxc.cast(evoxc.select(edge_valid, dst, zero), evoxc.int64),
            evoxc.cast(2, evoxc.int64),
        ),
        enp.full((pop_size, max_conns), I_INF, dtype=evoxc.int64),
    )
    node_key = evoxc.add(
        evoxc.multiply(evoxc.cast(slots, evoxc.int64), evoxc.cast(2, evoxc.int64)),
        evoxc.cast(1, evoxc.int64),
    )
    order = evoxc.argsort(
        evoxc.concatenate([edge_key, node_key], axis=1), axis=-1, stable=True
    )
    total = max_conns + max_nodes
    inverse = flat_scatter(
        _iinf_like(_node_slots(pop_size, total)),
        evoxc.cast(order, evoxc.int32),
        _node_slots(pop_size, total),
    )
    tail = evoxc.slice(inverse, (0, max_conns), (pop_size, max_nodes))
    return evoxc.subtract(tail, slots)


def incoming_table(rank, cc, dst, src, edge_valid, max_nodes, max_in_degree):
    """Pack each node's incoming edges into the static ``(P, N, K)`` table.

    Args:
        rank: ``(P, C)`` ``int32`` in-group rank from :func:`edge_group_rank`.
        cc: ``(P, N)`` ``int32`` cumulative in-degree from
            :func:`cum_in_degree`.
        dst: ``(P, C)`` ``int32`` destination slots.
        src: ``(P, C)`` ``int32`` source slots.
        edge_valid: ``(P, C)`` ``bool``.
        max_nodes (int): ``N``; trace-time static.
        max_in_degree (int): ``K``; trace-time static.

    Returns:
        tuple: ``(in_conn, in_src, edge_overflow)`` where ``in_conn`` is
        ``(P, N, K)`` ``int32`` (connection row index of the k-th incoming
        edge of node ``i``), ``in_src`` is ``(P, N, K)`` ``int32`` (that edge's
        source slot, ``I_INF`` in exactly the same positions) and
        ``edge_overflow`` is ``(P,)`` ``int32`` (number of valid edges dropped
        because their column reached ``K``).

    Raises:
        tensorneat.common.errors.ConfigError: If a static is not positive or
            disagrees with a tensor shape.
        evoxc.ShapeError: Propagated unchanged.

    Notes:
        ``start[i] = cc[i - 1]`` (``0`` for ``i == 0``) and
        ``col = rank - start[dst]``; the flat scatter target is
        ``dst * K + col`` into a ``(P, N * K + 1)`` buffer whose trailing cell
        is the dump slot, so edges with ``col >= K`` or with an invalid row go
        to the dump instead of throwing an out-of-range error. ``edge_overflow``
        is a ``reduce_sum`` of ``(col >= K) & edge_valid`` — data, never an
        exception (the host decides to rebuild or to raise
        :class:`~tensorneat.common.errors.ShapeBoundError`).
    """
    max_nodes = _positive_int(max_nodes, "max_nodes")
    max_in_degree = _positive_int(max_in_degree, "max_in_degree")
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
    column = evoxc.subtract(rank, flat_gather(start, safe_dst))
    kept_column = evoxc.select(edge_valid, column, zero)
    in_window = enp.logical_and(
        enp.greater_equal(kept_column, 0), enp.less(kept_column, max_in_degree)
    )
    keep = enp.logical_and(edge_valid, in_window)
    target = evoxc.add(
        evoxc.multiply(safe_dst, _scalar_int(max_in_degree)),
        evoxc.select(keep, kept_column, zero),
    )
    width = max_nodes * max_in_degree
    buffer_shape = (pop_size, width + 1)
    in_conn = evoxc.reshape(
        evoxc.slice(
            masked_flat_scatter(
                _iinf_like(enp.zeros(buffer_shape, dtype=evoxc.int32)),
                target,
                _node_slots(pop_size, max_conns),
                keep,
            ),
            (0, 0),
            (pop_size, width),
        ),
        (pop_size, max_nodes, max_in_degree),
    )
    in_src = evoxc.reshape(
        evoxc.slice(
            masked_flat_scatter(
                _iinf_like(enp.zeros(buffer_shape, dtype=evoxc.int32)),
                target,
                evoxc.select(edge_valid, src, _iinf_like(src)),
                keep,
            ),
            (0, 0),
            (pop_size, width),
        ),
        (pop_size, max_nodes, max_in_degree),
    )
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


def node_levels(in_conn, in_src, input_slot, max_nodes):
    """Longest-path depth of every node (the level-synchronous schedule).

    Args:
        in_conn: ``(P, N, K)`` ``int32`` incoming-edge table (``I_INF``
            padding); a padded column is never relaxed.
        in_src: ``(P, N, K)`` ``int32`` source slots of those edges.
        input_slot: ``(P, I)`` ``int32`` slots of the input nodes, pinned to
            level ``0``.
        max_nodes (int): ``N``; trace-time static, the slot count and the scale
            of the relaxation cap.

    Returns:
        ``(P, N)`` ``int32`` levels: ``0`` for input nodes, ``max(pred) + 1``
        otherwise, ``I_INF`` for an absent node, a node unreachable from the
        inputs, or a node on / downstream of a live cycle.

    Raises:
        tensorneat.common.errors.ConfigError: If ``max_nodes`` is not positive
            or disagrees with ``in_conn``.
        evoxc.ShapeError / evoxc.TraceError: Propagated unchanged.

    Notes:
        One ``evoxc.while_loop`` carrying **two** level relaxations over the
        same ``(P, N, K)`` window, each of the form
        ``x[i] = max(x[i], max_k(x[in_src[i, k]] + 1))`` with ``-1`` meaning
        "not settled":

        * ``level`` — seeded at ``0`` on the input slots (re-pinned every
          iteration) and ``-1`` elsewhere; its final value is the longest path
          from an input, i.e. the returned depth.
        * ``reach`` — seeded at ``0`` on **every** slot and never re-pinned; it
          is the longest *walk* ending at each node, so it grows without bound
          exactly for the nodes reachable from a cycle (a walk may start on the
          cycle itself, which is why this relaxation, unlike ``level``, is not
          frozen by a pinned input sitting on the cycle).

        A snapshot of both is taken after iteration ``N`` and the loop may run
        for at most ``2N`` iterations: a node bound for infinity has
        ``x(2N) >= 2N - (L - 1) >= N + 1 > N >= x(N)`` (``L`` = cycle length,
        ``L <= N``), while a node of a genuinely acyclic part has converged
        after at most ``N - 1`` iterations and cannot grow any more. Growth of
        ``reach`` between the two snapshots therefore marks exactly the nodes
        reachable from a cycle, which are reported as ``I_INF`` unless they are
        pinned inputs, and ``level`` still ``-1`` marks the nodes unreachable
        from the inputs.

        The loop carries only tensors of fixed shape/dtype and its condition is
        a scalar ``bool`` (a requirement of evoxc control flow); no ``vmap``
        may be applied over it — the batch axis is written explicitly, as
        everywhere in this package. Nothing is iterated per node in Python: the
        loop cost is ``O(N * K)`` per iteration, i.e. edge-proportional, and a
        feedforward population leaves the loop after ``depth`` iterations
        (``changed == False``); only a population that carries a reachable
        cycle pays the full ``2N``.
    """
    max_nodes = _positive_int(max_nodes, "max_nodes")
    pop_size, nodes, width = in_conn.shape
    pop_size, nodes, width = int(pop_size), int(nodes), int(width)
    if nodes != max_nodes:
        raise ConfigError(f"in_conn node axis {nodes} does not match max_nodes {max_nodes}")
    live = enp.not_equal(in_conn, _iinf_like(in_conn))
    # Every invalid window slot is routed to a trailing dump column holding
    # ``-2``: after the ``+1`` of the relaxation that is ``-1``, i.e. "this edge
    # contributes nothing", so neither the gathered block nor the reduction
    # needs an extra mask (``-1`` stays the "not settled yet" level).
    dump_index = enp.full((pop_size, nodes, width), nodes, dtype=evoxc.int32)
    index = evoxc.select(live, in_src, dump_index)
    dump = enp.full((pop_size, 1), -2, dtype=evoxc.int32)
    pinned = _pinned_mask(input_slot, _node_slots(pop_size, nodes))
    zero_level = enp.zeros((pop_size, nodes), dtype=evoxc.int32)
    minus_one = enp.full((pop_size, nodes), -1, dtype=evoxc.int32)
    one = _scalar_int(1)
    limit = 2 * max_nodes

    def relax(values):
        """One max-plus step of a longest-walk relaxation (``-1`` = not settled)."""
        source = flat_gather(evoxc.concatenate([values, dump], axis=1), index)
        best = evoxc.reduce_max(source, axes=(2,))
        # ``best < 0`` means "no settled predecessor" (dump or an unsettled
        # source), which must leave ``values`` untouched instead of settling
        # the node at ``0``.
        return evoxc.maximum(
            values,
            evoxc.select(enp.greater(best, -1), evoxc.add(best, one), minus_one),
        )

    def cond(state):
        step, _level, _reach, _level_at, _reach_at, changed = state
        return enp.logical_and(enp.less(step, limit), changed)

    def body(state):
        step, level, reach, level_at, reach_at, _changed = state
        next_level = evoxc.select(pinned, zero_level, relax(level))
        next_reach = relax(reach)
        next_step = evoxc.add(step, one)
        snapshot = enp.equal(next_step, max_nodes)
        moved = enp.logical_or(
            enp.not_equal(next_level, level), enp.not_equal(next_reach, reach)
        )
        changed = enp.greater(evoxc.reduce_max(evoxc.cast(moved, evoxc.int32)), 0)
        return (
            next_step,
            next_level,
            next_reach,
            evoxc.select(snapshot, next_level, level_at),
            evoxc.select(snapshot, next_reach, reach_at),
            changed,
        )

    _step, level, reach, _level_at, reach_at, _changed = evoxc.while_loop(
        cond,
        body,
        (
            enp.zeros((), dtype=evoxc.int32),
            evoxc.select(pinned, zero_level, minus_one),
            zero_level,
            _iinf_like(zero_level),
            _iinf_like(zero_level),
            enp.full((), True, dtype=evoxc.bool_),
        ),
    )
    reachable_from_cycle = enp.greater(reach, reach_at)
    unreachable = enp.less(level, 0)
    blank = enp.logical_and(
        enp.logical_not(pinned), enp.logical_or(reachable_from_cycle, unreachable)
    )
    return evoxc.select(blank, _iinf_like(level), level)


def _pinned_mask(input_slot, slots):
    """``(P, N)`` bool mask of the node slots named by ``input_slot``.

    Args:
        input_slot: ``(P, I)`` ``int32`` input slots (``I_INF`` allowed).
        slots: ``(P, N)`` ``int32`` node slot indices from :func:`_node_slots`.

    Returns:
        ``(P, N)`` bool tensor, ``True`` on every slot that carries an input.

    Notes:
        A broadcast equality over the ``I`` axis (cost ``O(N * I)``) instead of
        a scatter: ``I_INF`` never matches a real slot, so no index sanitizing
        and no dump column is needed.
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
        transformed (Transformed): A schedule produced by
            :func:`derive_schedule`, whose arrays have already been brought
            back to host memory (numpy). ``evoxc.Tensor`` leaves are accepted
            and read with ``.numpy()``.

    Returns:
        None: Returns ``None`` when every invariant of :class:`Transformed`
        holds.

    Raises:
        tensorneat.common.errors.ConfigError: Describing the *first* violated
            invariant, in this order — a shape or dtype that does not match the
            documented signature, ``in_conn``/``in_src`` disagreeing on a
            padded position, ``in_valid`` disagreeing with
            ``in_conn != I_INF``, ``topo_key`` not being ``I_INF`` exactly where
            ``node_level`` is, a negative ``edge_overflow``, or an
            ``input_slot``/``output_slot`` that points outside ``[0, N)`` and
            is not ``I_INF``.

    Notes:
        Host-only and deliberately not a graph function: it is the oracle the
        unit tests use against a numpy reference implementation of the
        derivation. It never repairs a schedule and never runs inside a
        generation, so it must stay free of any evoxc tracing. A schedule that
        reports ``edge_overflow > 0`` is *valid* here — lossy, but consistent.
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
    if not np.all(topo_key[node_level == I_INF] == I_INF):
        raise ConfigError(
            "validate_schedule: an I_INF node_level must pair with an I_INF topo_key"
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
    """Bring one schedule field to host memory as a numpy array."""
    if isinstance(value, np.ndarray):
        return value
    as_numpy = getattr(value, "numpy", None)
    if callable(as_numpy):
        return np.asarray(as_numpy())
    array = np.asarray(value)
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
