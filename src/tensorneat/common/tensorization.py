"""Tensorization: turn a ragged evolved topology into a static execution schedule.

Dependency DAG position: ``tensorneat.common`` bottom layer. Imports
:mod:`~tensorneat.common.tools` (flat-index helpers),
:mod:`~tensorneat.common.errors` and ``evoxc``; nothing from ``genome``,
``algorithm``, ``problem`` or ``pipeline``, and no ``jax``/``torch``. It replaces
the pre-rewrite ``common/graph.py``, whose topological sort ran over a dense
``(N, N)`` adjacency matrix at ``max_nodes ** 2`` cost.

What a schedule is
------------------
A schedule resolves every ragged connection row to ``(src, dst)`` **node slots**
(indices into that genome's own arrays, never innovation numbers), packs each
node's incoming edges into a ``(P, N, K)`` table and gives every node a
``node_level`` (longest-path depth) that orders a forward pass by level — work
proportional to real connections and depth, never to ``max_nodes ** 2``.

Output contract: :class:`Transformed` holds **arrays only** (no statics, no
callables), so it is a plain pytree that crosses a graph boundary and is cached
per shape signature; any static a consumer needs is implied by a shape or dtype.

Cost: ``O(P * (S + C log C + N * K * D))`` with ``S = key_span``,
``C = max_conns``, ``K = max_in_degree`` and ``D`` = network depth (``2N`` only
for a population with a live cycle), paid once per generation.

``evoxc`` constraints that shape this module: no ``cumsum`` (a StableHLO export
deferral, hence the ``cc`` counting trick), no scatter-with-reduce, no batched
``take_along_axis`` (hence flat indexing), no ``where`` (``evoxc.select``), and
out-of-range indices raise instead of clamping — every table index is sanitized
*and* masked. Everything here is in-graph: no host round-trip, no ``.numpy()``
and no Python ``if`` on a traced value (only on statics such as dtypes, shapes
and the declared static integers).
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
    """Static execution schedule of one population (all fields batch-first).

    Every field carries the population on axis 0; ``N = max_nodes``,
    ``C = max_conns``, ``K = max_in_degree``, ``I = len(input_keys)`` and
    ``O = len(output_keys)`` are static at build time. Sentinel: ``I_INF``
    (``int32`` max) means "no index / unmatched / unavailable"; ``NaN`` lives in
    the gene payload, never here. The twelve fields below are the contract and
    ``level`` / ``overflow`` are read-only compatibility properties.

    Invariances consumers may rely on:

    * ``in_conn``, ``in_src`` and ``in_valid`` mark the same positions
      (``in_valid == (in_conn != I_INF)``); a node's columns ``0..d-1`` are
      dense, ``d..K-1`` are padding, and ``in_conn[p, i, 0] == I_INF`` means
      "no incoming edge" (input or isolated node).
    * for every ``(p, c)`` with ``edge_valid[p, c]``, exactly one ``(p, i, k)``
      satisfies ``in_conn[p, i, k] == c`` and ``in_src[p, i, k] == src[p, c]``
      — except edges whose in-group rank reached ``K`` (``edge_overflow``, which
      also means the schedule is lossy: the host must rebuild with a larger
      ``max_in_degree``, never evaluate a truncated phenotype).
    * ``node_level[p, i]`` is ``I_INF`` iff node ``i`` is absent, unreachable
      from the inputs, or on / downstream of a live cycle;
      ``reduce_max(node_level)`` bounds the execution loop.
    * executing levels ``0, 1, ...`` in order is a valid topological order, and
      nodes within one level are independent.
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

    ``int32`` keys compare against ``I_INF``; ``float32`` keys are "not ``NaN``
    and below ``KEY_MAX``" (evoxc has no ``isfinite``). A trace-time branch on the
    static dtype: a negative or out-of-span key is dropped later by the ``[0, S)``
    test in :func:`_key_indices`, so both spellings agree on every real key.
    """
    if keys.dtype.kind in ("i", "u"):
        return enp.not_equal(keys, _iinf_like(keys))
    return enp.logical_and(
        enp.logical_not(evoxc.isnan(keys)),
        enp.less(keys, enp.full(tuple(keys.shape), KEY_MAX, dtype=keys.dtype)),
    )


def _narrow_int(limit):
    """Narrowest ``evoxc`` integer dtype whose range covers ``[0, limit]``.

    ``int8`` for ``limit <= 127``, ``int16`` for ``limit <= 32767``, else
    ``int32`` — a larger limit raises :class:`ConfigError` instead of silently
    truncating a sort key. ``numpy``'s stable ``argsort`` takes its radix path for
    narrow integers (on a ``(1000, 250)`` block: 0.6 / 1.0 / 7.4 ms for
    int8 / int16 / int32 keys), so the dtype comes from the static shape, never
    from the data.
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
    """Sanitized table indices of a key array plus the "indexable" mask.

    Returns ``(idx, in_range)``: ``int32`` indices of ``keys``' shape with every
    non-indexable entry replaced by ``0``, and the bool mask of the entries whose
    key is a real node key inside ``[0, key_span)``. A present key that does not
    fit ``key_span`` cannot be refused in-graph, so it reads as a miss — exactly
    like an unknown key — and an ``I_INF`` never reaches the backend. For
    ``int32`` keys the whole ``present & [0, S)`` test is one unsigned comparison
    (``cast(key, uint32) < S``), which also maps negative keys and ``I_INF``
    above ``S``; the ``float32`` spelling needs the explicit signed test.
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
        node_keys: ``(P, max_nodes)`` ``int32`` (``I_INF`` padding) or
            ``float32`` (``NaN`` padding) node **key column**; the dispatch is on
            the static dtype (see :func:`_key_present`).
        conn_in: ``(P, max_conns)`` source key of each connection row, same dtype
            and padding as ``node_keys``.
        conn_out: ``(P, max_conns)`` destination key, likewise.
        input_keys: ``(I,)`` keys of the input nodes, shared by the whole batch
            (no batch axis), same dtype as ``node_keys``.
        output_keys: ``(O,)`` keys of the output nodes, likewise shared.
        pop_size (int): ``P``; max_nodes (int): ``N``; max_conns (int): ``C``;
            max_in_degree (int): ``K``; key_span (int): ``S``, the key-magnitude
            bound of the ``(P, S + 1)`` key→slot table. All trace-time statics.
        acyclic (bool): ``False`` (default) keeps live-cycle detection, so a
            genome on or downstream of a cycle gets ``I_INF`` levels — but it pays
            for that only where a cycle exists: :func:`node_levels` runs the
            level-only relaxation first and a violation test sends the population
            through the cycle-detecting path inside ``evoxc.cond`` only when some
            member holds a cycle. ``True`` asserts that the population is
            cycle-free and drops both, so a cyclic genome then gets a saturated
            level instead of ``I_INF``; a feed-forward genome may pass ``True``, a
            recurrent one may not.

    The statics are positional-or-keyword (not keyword-only), because
    ``evoxc.build`` mirrors a positional call while
    ``functools.partial(derive_schedule.fn, **statics)`` must keep working.

    Returns:
        Transformed: the schedule described by :class:`Transformed` — arrays
        only, so the pytree can cross a graph boundary and be cached per shape.

    Raises:
        tensorneat.common.errors.ConfigError: At trace time, for a non-positive
            or shape-inconsistent static, or for key arrays of differing dtypes.

    Notes:
        Composition; every step is a separate, individually testable helper:

        1. :func:`key_slot_table` — dense key→slot table, trailing dump column.
        2. :func:`slot_lookup` — ``src``/``dst`` from that table; an unmatched
           endpoint is ``I_INF`` and makes the row invalid (a present key no node
           holds therefore needs no separate ``present`` test).
        3. :func:`edge_order` — the global in-edge rank of every row and the
           ``cumsum``-free per-node prefix count, from **one** stable sort (the
           fused form of :func:`edge_group_rank` + :func:`cum_in_degree`, which
           stay available for separate use).
        4. :func:`incoming_table` — ``in_conn``/``in_src``/``edge_overflow`` from
           ``col = rank - start[dst]``; ``in_valid`` is then free.
        5. :func:`node_levels` — longest-path relaxation in a ``while_loop``,
           inputs pinned to level 0, cycles reported as ``I_INF`` unless
           ``acyclic`` asserts there are none; the cycle-detecting relaxation runs
           only where :func:`_cycle_violation` finds an edge the levels
           contradict.
        6. ``node_exists``, ``topo_key``, ``input_slot`` and ``output_slot`` are
           then free (``key != I_INF``/not ``NaN``, ``level * N + slot``, and two
           lookups of the shared key vectors).

        Value validation is impossible inside a trace: a key outside
        ``[0, key_span)`` and an input/output key no node holds both surface as
        ``I_INF`` slots (a miss) instead of raising. The host that owns the
        payload refuses such a payload; this kernel reports, it never guesses.
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
        node_keys: ``(P, N)`` ``int32`` (``I_INF`` padding) or ``float32``
            (``NaN`` padding) node key column, keys integer-valued in
            ``[0, key_span)``.
        key_span (int): ``S``, the static key-magnitude bound.

    Returns:
        ``(P, S + 1)`` ``int32`` table: ``table[p, k]`` is the slot of genome
        ``p``'s node with key ``k``, ``I_INF`` when no node holds it. Column ``S``
        is the **dump slot**: absent rows are written there instead of index
        ``0``, which is what makes one unmasked ``flat_scatter`` safe.

    Raises:
        tensorneat.common.errors.ConfigError: If ``key_span`` is not positive.

    Notes:
        One :func:`~tensorneat.common.tools.masked_flat_scatter` of the slot
        indices, masked by "the key is real and inside ``[0, S)``"; a key outside
        that range cannot be refused in-graph, so it lands in the dump slot and
        reads as a miss. Under the float spelling a key at or above ``2**24``
        (``tools.KEY_MAX``) is also absent, being no longer ``float32``-exact.
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
        keys: ``(P, L)`` key array to resolve (connection endpoints, or a
            broadcast shared input/output key vector), same dtype as the table;
            ``NaN``/``I_INF`` allowed and always a miss.
        key_span (int): ``S``, matching the table.

    Returns:
        ``(P, L)`` ``int32`` slots in ``[0, N)``, ``I_INF`` where the key is the
        padding sentinel, unknown, or outside ``[0, S)``.

    Raises:
        tensorneat.common.errors.ConfigError: If ``table``'s second dimension is
            not ``key_span + 1``.

    Notes:
        Indices are made safe *before* the gather (sentinel → ``0``, then a range
        test) because evoxc raises on out-of-range gathers; the gather's own mask
        then overwrites the miss cases with ``I_INF``, so the sanitized ``0``
        never leaks into the result. Every selected index is a sanitized key
        inside ``[0, key_span)`` and the table is ``key_span + 1`` wide, so the
        index block is built ``trusted`` (no per-element range test).
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

    Private body of :func:`slot_lookup` (no width check), also called by
    :func:`derive_schedule` for a table built by :func:`key_slot_table`.
    """
    indices, in_range = _key_indices(keys, key_span)
    flat_index = prepare_flat_index(table, indices, in_range, 0, trusted=True)
    values = flat_gather_prepared(table, flat_index, indices.shape)
    return evoxc.select(in_range, values, _scalar_constant(I_INF, evoxc.int32))


def edge_order(dst, edge_valid, max_nodes, max_conns):
    """In-edge rank of every connection row plus per-node in-degree prefixes.

    Args:
        dst: ``(P, C)`` ``int32`` destination slots; ``I_INF`` where the row is
            absent or unmatched.
        edge_valid: ``(P, C)`` ``bool`` from the lookup.
        max_nodes (int): ``N``; max_conns (int): ``C``.

    Returns:
        tuple: ``(rank, cc)``. ``rank`` is ``(P, C)`` ``int32``: the position of
        row ``c`` among its genome's valid rows ordered by ``(dst, connection
        index)`` — the number of valid rows before it — with ``I_INF`` on an
        invalid row. ``cc`` is ``(P, N)`` ``int32``: the number of valid rows
        whose destination slot is ``<= i``, so node ``i``'s in-edge group starts
        at ``cc[i - 1]`` (``0`` for ``i == 0``) and an edge's in-group column is
        ``rank - start[dst]``.

    Raises:
        tensorneat.common.errors.ConfigError: If a static is not positive or
            disagrees with ``dst``'s width.

    Notes:
        **One** stable ``argsort`` over the ``C + N`` combined entries — the ``C``
        edge rows keyed ``2 * dst`` and the ``N`` node slots keyed ``2 * slot +
        1`` — yields both answers from its inverse permutation ``inv``:
        ``rank[c] = inv[c] - dst[c]`` (exactly ``dst[c]`` node entries sort below
        an edge at slot ``dst[c]``, and same-destination edges keep connection
        order through ``stable=True``) and ``cc[i] = inv[C + i] - i``. This is
        the fused form of :func:`edge_group_rank` + :func:`cum_in_degree`, which
        stay available separately but each pay for their own sort.

        Keys never leave ``[0, 2 * N + 2]``: an edge key precedes its own node
        key so an edge counts as ``"<= i"``, and an invalid row is keyed ``2 * N
        + 2``, one past the last node key, so it sorts after everything without
        spending ``I_INF`` (which no narrow integer can hold). That bound is what
        lets :func:`_narrow_int` pick a radix-sortable carrier: ``int8`` for
        ``N <= 62``, ``int16`` for ``N <= 16382``, ``int32`` above.
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
        dst: ``(P, C)`` ``int32`` destination slots; ``I_INF`` where the row is
            absent or unmatched.
        edge_valid: ``(P, C)`` ``bool`` from the lookup.
        max_conns (int): ``C``.

    Returns:
        ``(P, C)`` ``int32``: ``rank[p, c]`` = number of valid rows with a
        smaller destination slot, plus the number of valid rows with the same
        ``dst`` and a smaller connection index — i.e. the position of row ``c``
        among the valid rows ordered by ``(dst, connection index)``.
        ``I_INF`` for invalid rows.

    Raises:
        tensorneat.common.errors.ConfigError: If ``max_conns`` is not positive or
            disagrees with ``dst``'s width.

    Notes:
        One ``argsort(..., stable=True)`` over the ``int32`` composite key
        ``dst * C + conn_index``, then one ``flat_scatter`` of ``arange(C)`` to
        invert the permutation. Invalid rows are sanitized to slot ``0`` *before*
        the composite is built, so every real key is below ``N * C`` and an
        invalid row can be keyed ``I_INF`` ("after every valid row") without
        overflow, at half the ``int64`` sort traffic. ``stable=True`` keeps
        connection-row order inside a ``dst`` group, which makes the retained
        window deterministic.

        :func:`derive_schedule` does not call this: :func:`edge_order` returns
        the same ``rank`` from the sort it needs for ``cc`` anyway, with a key
        narrow enough for the radix path. This standalone form cannot narrow its
        key, because the composite is bounded by ``max_nodes * max_conns`` and
        ``max_nodes`` is not an argument here.
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
        dst: ``(P, C)`` ``int32`` destination slots (``I_INF`` for invalid rows).
        edge_valid: ``(P, C)`` ``bool``.
        max_nodes (int): ``N``.

    Returns:
        ``(P, N)`` ``int32``: ``cc[p, i]`` = number of valid edges of genome ``p``
        whose destination slot is ``<= i``. Node ``i``'s in-edge group starts at
        ``cc[i - 1]`` (``0`` for ``i == 0``), so an edge's column is
        ``rank - start[dst]``.

    Raises:
        tensorneat.common.errors.ConfigError: If ``max_nodes`` is not positive.

    Notes:
        Deliberately ``cumsum``-free (``cumsum`` is a StableHLO export deferral in
        evoxc, numpy-interpreter only): the merged stable ``argsort`` of ``C``
        edge entries keyed ``2 * dst`` and ``N`` node entries keyed ``2 * slot +
        1`` yields ``cc[i] = inverse[C + i] - i``. This is the ``cc`` half of
        :func:`edge_order`, which :func:`derive_schedule` calls once for both
        outputs; here the rank half is simply discarded.
        """
    max_nodes = _positive_static(max_nodes, "max_nodes")
    _rank, cc = edge_order(dst, edge_valid, max_nodes, int(dst.shape[1]))
    return cc


def incoming_table(rank, cc, dst, src, edge_valid, max_nodes, max_in_degree):
    """Pack each node's incoming edges into the static ``(P, N, K)`` table.

    Args:
        rank: ``(P, C)`` ``int32`` in-edge rank from :func:`edge_order`.
        cc: ``(P, N)`` ``int32`` cumulative in-degree from :func:`edge_order`.
        dst: ``(P, C)`` ``int32`` destination slots; src: source slots.
        edge_valid: ``(P, C)`` ``bool``.
        max_nodes (int): ``N``; max_in_degree (int): ``K``.

    Returns:
        tuple: ``(in_conn, in_src, edge_overflow)`` — ``(P, N, K)`` ``int32``
        connection row indices, ``(P, N, K)`` ``int32`` source slots (``I_INF``
        in the same positions) and ``(P,)`` ``int32`` counts of valid edges whose
        column reached ``K``.

    Raises:
        tensorneat.common.errors.ConfigError: If a static is not positive or
            disagrees with a tensor shape.

    Notes:
        ``start[i] = cc[i - 1]`` (``0`` for ``i == 0``) and
        ``col = rank - start[dst]``; the flat scatter target is ``dst * K + col``
        into a ``(P, N * K + 1)`` buffer whose trailing cell is the dump slot, so
        edges with ``col >= K`` or an invalid row go to the dump instead of
        throwing an out-of-range error. ``edge_overflow`` is a ``reduce_sum`` of
        ``(col >= K) & edge_valid`` — data, never an exception (the host decides
        to rebuild or to raise ``ShapeBoundError``).

        The two scatter passes (the ``in_conn`` row index and the ``in_src``
        source slot) share **one** index block (:func:`_window_block`) because
        only their update payloads differ, and both index sources are provably
        in range: ``safe_dst`` is a node slot, ``in_src``'s padded ``I_INF``
        entries are dropped writes onto the dump cell.
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
    """Flat scatter block into the ``(P, N*K + 1)`` window buffer, built once.

    ``target`` is ``dst * K + col`` (``col < K``, ``dst < N``, so provably inside
    ``[0, N*K)``) on every row with ``keep``; a dropped row is routed to the dump
    cell ``N*K``. Both bounds are constructive, so the block is ``trusted``.
    """
    width = max_nodes * max_in_degree
    return prepare_flat_index((pop_size, width + 1), target, keep, width, trusted=True)


def _window_table(block, updates, pop_size, max_nodes, max_in_degree):
    """Scatter one payload into a fresh ``(P, N, K)`` window table.

    The shared body of :func:`incoming_table`'s two scatter passes: an
    ``(P, N*K + 1)`` ``I_INF`` buffer written through ``block`` (dropped rows land
    in the trailing dump cell), the dump column sliced off and reshaped to the
    window. ``updates`` is the row index (``in_conn``) or the source slot
    (``in_src``); its value on a row routed to the dump cell is ignored.
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
            only its shape is read here.
        in_src: ``(P, N, K)`` ``int32`` source slots of those edges, ``I_INF`` in
            exactly the padded positions — those route to the dump column.
        input_slot: ``(P, I)`` ``int32`` slots of the input nodes, pinned to 0.
        max_nodes (int): ``N``; the slot count and the scale of the loop cap.
        acyclic (bool): ``False`` (default) keeps the live-cycle semantics and pays
            for them only where a cycle exists: the level-only relaxation runs
            first, :func:`_cycle_violation` tests its output, and the
            cycle-detecting fallback runs inside ``evoxc.cond`` only for a
            population that fails that test. ``True`` asserts that the population
            is cycle-free and drops test and fallback, so a cyclic genome then
            receives a saturated (wrong) level instead of ``I_INF``.
        edges: ``(src, dst, edge_valid)`` — the ``(P, C)`` endpoint slots and the
            "both endpoints matched" mask of the connection rows; required when
            ``acyclic`` is ``False``, being the fallback's trigger.

    Returns:
        ``(P, N)`` ``int32`` levels: ``0`` for input nodes, ``max(pred) + 1``
        otherwise, ``I_INF`` for an absent node, a node unreachable from the
        inputs, or (only with ``acyclic=False``) a node on / downstream of a cycle.

    Raises:
        tensorneat.common.errors.ConfigError: If a static is not positive or
            disagrees with a tensor shape, or if ``acyclic`` is ``False`` without
            the ``edges`` triple.

    Notes:
        A max-plus relaxation ``x[i] = max(x[i], max_k(x[in_src[i, k]] + 1))`` in an
        ``evoxc.while_loop``, with ``-1`` = "not settled" and ``-2`` the dump cell's
        value (so an unsettled predecessor is neutral and no mask is needed). The
        window's flat index block depends only on ``in_src`` and is built once
        outside the loop (:func:`~tensorneat.common.tools.prepare_flat_index`); an
        iteration concatenates its ``(P, N)`` state to a one-cell ``-2`` dump
        column, gathers through that block and folds the ``K`` candidates pairwise
        (0.30 ms against 1.26 ms for one ``reduce_max`` over the trailing axis at
        ``P=1000, N=50, K=8``).

        ``level`` is seeded at ``0`` on the input slots, re-pinned every iteration,
        and at ``-1`` elsewhere. The fallback relaxes a second state, ``reach``: the
        longest *walk* ending at each node, seeded at ``0`` on every slot and never
        re-pinned, so it grows without bound exactly on the nodes reachable from a
        cycle (a walk may start on the cycle, which is why a pinned input does not
        freeze it). Snapshotting it after iteration ``N`` of a loop that may run
        ``2N`` separates those nodes (growth between the snapshots) from the
        acyclic ones, which converged within ``N - 1``; a ``level`` still ``-1``
        marks a node unreachable from the inputs.

        Both campaigns are one ``settle(cap, detect_cycles)``; the level-only one
        leaves after ``depth`` iterations (``changed`` turns false) at ``O(N * K)``
        per iteration, and only the fallback can reach its cap.
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

    A relaxation value lies in ``[-2, cap]`` (``-2`` the dump cell, ``0`` a pinned
    input, ``cap`` the largest level or walk a capped loop can reach), so ``int8``
    serves ``cap <= 127`` and ``int16`` serves ``cap < 32767``; the choice follows
    the static shape, never the data. ``no_snapshot`` is the cycle-detecting
    campaign's "no snapshot taken yet" sentinel and must sit above every reachable
    walk length (``I_INF`` for ``int32``, where no larger value exists).
    """
    if cap <= 127:
        return evoxc.int8, 127
    if cap < 32767:
        return evoxc.int16, 32767
    return evoxc.int32, I_INF


def _cycle_violation(level, src, dst, edge_valid, max_nodes):
    """0-d bool: some member has an edge whose settled levels are not increasing.

    Args:
        level: ``(P, N)`` levels of the level-only relaxation, **before** the
            ``I_INF`` mapping — i.e. ``-1`` for "never settled" and ``0 .. N``
            elsewhere; any integer dtype.
        src: ``(P, C)`` ``int32`` source slots, ``I_INF`` on an absent row or an
            unmatched key; dst: the destination slots, likewise.
        edge_valid: ``(P, C)`` ``bool`` — the row is a real connection and both of
            its endpoint keys matched a node.
        max_nodes (int): ``N``.

    Returns:
        A 0-d ``bool``: ``True`` iff at least one member has a valid edge ``(u, v)``
        with ``level[u] >= level[v]``.

    Notes:
        The exactness test of the ``acyclic=False`` fast path. A member whose levels
        settled satisfies ``level[v] = max(level[preds]) + 1 > level[u]`` on every
        edge, and every cycle breaks it: a reachable one grows until the cap, one
        through a pinned input leaves its non-pinned successors growing, and an
        unreachable one leaves both endpoints at ``-1``. A member with an
        *unreachable edge* (a connection whose source never settles) violates it
        too, which costs a fallback and never a wrong answer.

        Two ``(P, C)`` gathers over the ``(P, N + 1)`` level vector: a slot is
        clamped to the trailing ``-2`` dump cell, so an ``I_INF`` endpoint cannot
        index out of range, and ``edge_valid`` masks the half-matched rows the dump
        value would otherwise read as a violation. One ``reduce_max`` per member,
        then one over the population — data for ``evoxc.cond``, never Python control
        flow.
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

    A broadcast equality over the ``I`` axis (``O(N * I)``) instead of a scatter:
    ``I_INF`` never matches a real slot, so no index sanitizing or dump column.
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
            :func:`derive_schedule`, whose arrays have already been brought back
            to host memory (numpy). ``evoxc.Tensor`` leaves are accepted and read
            with ``.numpy()``.

    Returns:
        None: Returns ``None`` when every invariant of :class:`Transformed` holds.

    Raises:
        tensorneat.common.errors.ConfigError: Describing the *first* violated
            invariant, in this order — a shape or dtype outside the documented
            signature, ``in_conn``/``in_src`` disagreeing on a padded position,
            ``in_valid`` disagreeing with ``in_conn != I_INF``,
            ``topo_key``/``node_level`` disagreeing, a negative
            ``edge_overflow``, or an ``input_slot``/``output_slot`` outside
            ``[0, N)``.
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
    """Bring one schedule field to host memory as a numpy array.

    The conversion itself is :func:`~tensorneat.common.tools.to_numpy` (the one
    host-side accessor of this package); this wrapper only adds the
    validation-specific refusal of a non-array leaf.
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
