"""Batched, in-graph genome-array editing plus host-only renumbering.

Every kernel here is batch-first (population on axis 0) with static shapes, so a whole
population is edited by one elementwise/``gather``/``scatter``/``reduce`` call: no Python
loop over individuals, no ``vmap``, no ``scan``/``while_loop``/``cond``.
Absent rows are ``I_INF`` (``int32`` max) in ``*_keys`` and NaN in ``*_attrs``; the two
sentinels never mix.  ``re_cound_idx`` is host-only: it converts with ``Tensor.numpy()``
and never feeds its result back into a graph.

Every write here goes through ``evoxc``'s flattened-index helpers
(:func:`~tensorneat.common.tools.flat_scatter`,
:func:`~tensorneat.common.tools.masked_flat_gather`,
:func:`~tensorneat.common.tools.masked_flat_scatter`), because ``evoxc.gather``/
``evoxc.scatter`` replace the indexed axis instead of indexing it per row, and the numpy
backend *raises* on an out-of-range index instead of clamping.  ``I_INF`` is therefore
never handed to the backend: a "no write" row is redirected to a real row and the caller
gates the result with ``select``.
"""

from typing import NamedTuple

import evoxc
import evoxc.numpy as enp
import numpy as np

from ..common import (
    I_INF,
    ConfigError,
    fetch_first,
    fetch_random,
    flat_scatter,
    masked_flat_gather,
    masked_flat_scatter,
    to_numpy,
)


class GenomeTensors(NamedTuple):
    """The four payload arrays of a batch of genomes, population on axis 0.

    ``P`` is the population size, ``N`` ``max_nodes``, ``E`` ``max_conns``; ``NA``/``CK``/
    ``CA`` are ``node_gene.attr_width``, ``conn_gene.key_width`` and
    ``conn_gene.attr_width``.  All four shapes are static at trace time, so a change forces
    a rebuild of the cached executable.

    Row/key invariant: ``initialize`` writes nodes in key order, so the input keys occupy
    rows ``0..num_inputs-1`` and the output keys occupy rows
    ``max_nodes-num_outputs..max_nodes-1``.  Those rows are never deleted or moved, which is
    what makes ``get_input_idx()``/``get_output_idx()`` usable as *row indices*.  Hidden
    node keys are arbitrary and may move rows, so graph identity is compared on keys, never
    on row positions.

    Attributes:
        node_keys: ``(P, N) int32`` node key per row, ``I_INF`` for an absent row.
        node_attrs: ``(P, N, NA) float32`` node attribute rows, NaN for an absent row.
        conn_keys: ``(P, E, CK) int32`` ``(in_key, out_key)`` (plus ``historical_marker``
            when the conn gene declares it), ``I_INF`` for an absent row.
        conn_attrs: ``(P, E, CA) float32`` connection attribute rows (e.g. ``weight``),
            NaN for an absent row.
    """

    node_keys: evoxc.SymbolicTensor
    node_attrs: evoxc.SymbolicTensor
    conn_keys: evoxc.SymbolicTensor
    conn_attrs: evoxc.SymbolicTensor

    @property
    def field_names(self) -> tuple[str, ...]:
        """Field names in payload order, for generic payload walks.

        Returns:
            ``("node_keys", "node_attrs", "conn_keys", "conn_attrs")``.
        """
        return self._fields


# --------------------------------------------------------------------------- #
# private plumbing: int32 constants, key columns, per-row writes
# --------------------------------------------------------------------------- #


def _i32(value: int) -> evoxc.SymbolicTensor:
    """Rank-0 ``int32`` constant holding ``value``.

    Written out instead of a bare Python literal because ``int32`` tensor + Python ``int``
    promotes to ``int64`` in ``evoxc`` (plain ``np.result_type``), which would silently
    widen every index computation here.

    Args:
        value: Static integer baked into the trace.

    Returns:
        Rank-0 ``int32`` tensor inside the active trace.
    """
    return enp.full((), value, dtype=evoxc.int32)


def _key_column(kind_keys: evoxc.SymbolicTensor) -> evoxc.SymbolicTensor:
    """First key column of one gene family, ``(P, M)`` whatever its row width is.

    Args:
        kind_keys: ``(P, M)`` node keys, or ``(P, M, CK)`` connection keys.

    Returns:
        ``(P, M)`` ``int32`` column: both families store their primary key first
        (``in_key`` for a connection), so the ``I_INF`` "absent row" test reads the same
        column for either width.
    """
    if len(kind_keys.shape) == 3:
        return kind_keys[:, :, 0]
    return kind_keys


def _safe_rows(rows: evoxc.SymbolicTensor, pop_size: int) -> evoxc.SymbolicTensor:
    """Redirect the ``I_INF`` ("no write") row to row ``0`` so the backend never raises.

    Args:
        rows: ``(P,) int32`` caller-chosen row per individual.
        pop_size: Population size ``P`` (static).

    Returns:
        ``(P,) int32`` rows in ``[0, M)``: ``I_INF`` becomes ``0``, everything else is
        passed through (an out-of-range row still reaches the backend and raises there,
        because that is a caller bug, not a sentinel).
    """
    return evoxc.select(
        evoxc.equal(rows, _i32(I_INF)), _i32(0), enp.reshape(rows, (pop_size,))
    )


def _write_rows(
    x: evoxc.SymbolicTensor,
    rows: evoxc.SymbolicTensor,
    values: evoxc.SymbolicTensor,
) -> evoxc.SymbolicTensor:
    """Overwrite caller-chosen rows of one batch-first payload array.

    The write is unconditional at a row: ``I_INF`` is redirected to row ``0`` rather than
    skipped, because no ``scatter`` can be a per-row no-op.  Callers therefore gate the
    result with ``select`` (``.operations.mutation`` does exactly that).

    Args:
        x: ``(P, M)`` key array or ``(P, M, W)`` attribute array.
        rows: ``(P,) int32`` destination row per individual, ``I_INF`` for "no write".
        values: ``(P,)`` for a 2-D payload, ``(P, W)`` for a 3-D one, of ``x``'s dtype.

    Returns:
        ``x`` with ``rows`` overwritten, same shape and dtype.

    Raises:
        evoxc.ShapeError / evoxc.BackendError: Propagated unchanged for a row outside
            ``[0, M)`` or a dtype mismatch between ``x`` and ``values``.

    Notes:
        The caller's gate must be reshaped to ``x``'s rank before ``evoxc.select``
        (``(P, 1)`` for a key array, ``(P, 1, 1)`` for an attribute array):
        ``select`` broadcasts operands but not a lower-rank predicate.
    """
    pop_size, row_count = x.shape[0], x.shape[1]
    safe = _safe_rows(rows, pop_size)
    if len(x.shape) == 2:
        return flat_scatter(x, enp.reshape(safe, (pop_size, 1)), enp.reshape(values, (pop_size, 1)))
    width = x.shape[2]
    # Flat index of every (row, column) cell of the 3-D payload: row * W + column.
    row_offsets = evoxc.gather(
        enp.arange(0, row_count * width, width, dtype=evoxc.int32), safe, axis=0
    )
    columns = enp.reshape(row_offsets, (pop_size, 1)) + enp.arange(0, width, dtype=evoxc.int32)
    flat = enp.reshape(x, (pop_size, row_count * width))
    return enp.reshape(flat_scatter(flat, columns, values), (pop_size, row_count, width))


def _node_rows(
    query_keys: evoxc.SymbolicTensor,
    node_keys: evoxc.SymbolicTensor,
    node_present: evoxc.SymbolicTensor,
) -> tuple[evoxc.SymbolicTensor, evoxc.SymbolicTensor]:
    """Map each key of a query block to the row of the node that carries it.

    Node keys are arbitrary and a node may sit on any row, so a key is resolved to a row by
    matching against ``node_keys`` — the one ``O(N)``-wide pass :func:`check_cycles` pays
    once per edge endpoint, never per hop.

    Args:
        query_keys: ``(P,)`` or ``(P, E)`` ``int32`` node keys to look up.
        node_keys: ``(P, N) int32`` node key per row, ``I_INF`` padded.
        node_present: ``(P, N) bool`` row-validity mask (``node_keys != I_INF``).

    Returns:
        ``(rows, hit)``: ``(P,)``/``(P, E)`` ``int32`` row indices (``0`` where no node
        carries the key — the caller must gate on ``hit``, never on the raw row) and the
        matching ``(P,)``/``(P, E)`` bool hit mask, both shaped like ``query_keys``.

    Notes:
        The comparison is written in three dimensions (``(P, 1, 1)`` or ``(P, E, 1)``
        against ``(P, 1, N)``) because both call sites need one broadcast per query
        position and ``evoxc`` has no ``newaxis`` indexing — ``reshape`` is the only way to
        add an axis, and a two-dimensional query must be padded to rank three or the
        right-aligned broadcast would line it up against the node axis.
    """
    shape = tuple(query_keys.shape)
    match = evoxc.logical_and(
        evoxc.equal(
            enp.reshape(query_keys, shape + (1,) * (3 - len(shape))),
            enp.reshape(node_keys, (node_keys.shape[0], 1, node_keys.shape[1])),
        ),
        enp.reshape(node_present, (node_keys.shape[0], 1, node_keys.shape[1])),
    )
    hit = enp.reshape(evoxc.reduce_max(match, axes=-1), shape)
    rows = enp.reshape(
        evoxc.cast(evoxc.argmax(evoxc.cast(match, evoxc.int32), axis=-1), evoxc.int32), shape
    )
    return rows, hit


# --------------------------------------------------------------------------- #
# row bookkeeping
# --------------------------------------------------------------------------- #


def valid_count(kind_keys: evoxc.SymbolicTensor) -> evoxc.SymbolicTensor:
    """Count the existing rows of one gene family, per individual.

    Args:
        kind_keys: ``(P, M) int32`` key rows of a node *or* connection family (only the
            leading key column is inspected, so a ``(P, M, CK)`` connection key array is
            accepted too); ``I_INF`` marks an absent row.

    Returns:
        ``(P,) int32`` number of rows whose first key column is not ``I_INF``.

    Notes:
        The reduce returns ``int64`` in ``evoxc`` whatever the input dtype, so the result is
        cast back to ``int32`` to keep every downstream index computation narrow.
    """
    present = evoxc.not_equal(_key_column(kind_keys), _i32(I_INF))
    counted = evoxc.reduce_sum(evoxc.cast(present, evoxc.int32), axes=1)
    return evoxc.cast(counted, evoxc.int32)


def first_free_row(kind_keys: evoxc.SymbolicTensor) -> evoxc.SymbolicTensor:
    """Locate the row a new gene of one family would be written to.

    Args:
        kind_keys: ``(P, M) int32`` key rows of one gene family, ``I_INF`` padded.

    Returns:
        ``(P,) int32`` first row whose first key column is ``I_INF``, or ``I_INF`` when the
        family is full (``M`` existing rows).

    Notes:
        Delegates to :func:`~tensorneat.common.tools.fetch_first`, i.e. ``argmax`` of the
        free-row mask plus a mask-check that maps "no free row" to ``I_INF``.  ``argmax``
        over an all-False mask would silently return row ``0`` — a real node — which is
        exactly the sentinel mix this helper prevents.
    """
    free = evoxc.equal(_key_column(kind_keys), _i32(I_INF))
    # ``_key_column`` narrows a ``(P, M, CK)`` connection table to its ``(P, M)`` key
    # column, so the connection family is served by the same call.
    return fetch_first(free)


def fetch_random_index(
    key: evoxc.SymbolicTensor, mask: evoxc.SymbolicTensor
) -> evoxc.SymbolicTensor:
    """Draw one uniformly chosen True index per row of a mask.

    ``evoxc`` has no ``choice`` op (and no exportable ``cumsum``), so the draw is a
    ``(P, M)`` block of i.i.d. uniforms whose masked-out positions are set to ``-inf``
    before an ``argmax``: the maximiser of i.i.d. draws restricted to a set is uniform over
    that set, which makes the choice uniform over the True entries of each row.

    Args:
        key: Rank-0 ``evoxc.random`` key; the single draw covers the whole batch, so keys
            are never split per individual.
        mask: ``(P, M) bool`` selection mask, one row per individual.

    Returns:
        ``(P,) int32`` chosen column index, ``I_INF`` where the row has no True entry.

    Notes:
        Delegates to :func:`~tensorneat.common.tools.fetch_random`; ``I_INF`` (never the
        ``argmax`` fallback column) is what tells the caller "nothing to choose", because
        column ``0`` is a real gene.
    """
    return fetch_random(key, mask)


# --------------------------------------------------------------------------- #
# structural edits
# --------------------------------------------------------------------------- #


def add_node(
    node_keys: evoxc.SymbolicTensor,
    node_attrs: evoxc.SymbolicTensor,
    rows: evoxc.SymbolicTensor,
    new_keys: evoxc.SymbolicTensor,
    new_attrs: evoxc.SymbolicTensor,
) -> tuple[evoxc.SymbolicTensor, evoxc.SymbolicTensor]:
    """Write new node rows at caller-chosen free rows of every individual.

    The write is unconditional at ``rows``, so a caller must gate the result with ``select``
    when ``rows == I_INF`` for some individual (as ``.operations.mutation`` does); garbage
    is written at such a row instead of the write being skipped, because no scatter can be a
    no-op per row.  The ``(P,)`` gate has to be reshaped to the payload rank first — see
    :func:`_write_rows`.

    The row/key invariant is the caller's business: it passes ``first_free_row``, which only
    ever names a row that is already ``I_INF``, so the input rows
    (``0..num_inputs-1``) and the output rows (``max_nodes-num_outputs..max_nodes-1``) are
    never candidates.  Keys and attributes are written in the same row layout, so a payload
    stays consistent for the next ``transform``.

    Args:
        node_keys: ``(P, N) int32`` node key rows, ``I_INF`` padded.
        node_attrs: ``(P, N, NA) float32`` node attribute rows, NaN padded.
        rows: ``(P,) int32`` destination row per individual (``first_free_row``), ``I_INF``
            to mean "no write".
        new_keys: ``(P,) int32`` caller-allocated, population-unique node keys; never drawn
            from the RNG argument.
        new_attrs: ``(P, NA) float32`` attribute rows of the new nodes.

    Returns:
        ``(node_keys, node_attrs)`` with the new rows written at ``rows``.

    Raises:
        evoxc.ShapeError / evoxc.BackendError: Propagated unchanged (row out of range,
            dtype mismatch).
    """
    return (
        _write_rows(node_keys, rows, new_keys),
        _write_rows(node_attrs, rows, new_attrs),
    )


def delete_node_by_pos(
    node_keys: evoxc.SymbolicTensor,
    node_attrs: evoxc.SymbolicTensor,
    rows: evoxc.SymbolicTensor,
) -> tuple[evoxc.SymbolicTensor, evoxc.SymbolicTensor]:
    """Delete node rows by row position, restoring the padding sentinels.

    Args:
        node_keys: ``(P, N) int32`` node key rows.
        node_attrs: ``(P, N, NA) float32`` node attribute rows.
        rows: ``(P,) int32`` row to clear per individual, ``I_INF`` to mean "no delete".

    Returns:
        ``(node_keys, node_attrs)`` with key row ``rows`` set to ``I_INF`` and attribute row
        ``rows`` set to NaN; input and output rows must never be passed here.

    Raises:
        evoxc.ShapeError / evoxc.BackendError: Propagated unchanged (row out of range).
    """
    pop_size, attr_width = node_keys.shape[0], node_attrs.shape[2]
    return (
        _write_rows(node_keys, rows, enp.full((pop_size,), I_INF, dtype=node_keys.dtype)),
        _write_rows(
            node_attrs, rows, enp.full((pop_size, attr_width), float("nan"), dtype=node_attrs.dtype)
        ),
    )


def add_conn(
    conn_keys: evoxc.SymbolicTensor,
    conn_attrs: evoxc.SymbolicTensor,
    rows: evoxc.SymbolicTensor,
    new_keys: evoxc.SymbolicTensor,
    new_attrs: evoxc.SymbolicTensor,
) -> tuple[evoxc.SymbolicTensor, evoxc.SymbolicTensor]:
    """Write new connection rows at caller-chosen free rows of every individual.

    Same unconditional-write contract as :func:`add_node`: garbage is written when
    ``rows == I_INF`` and the caller gates the result with ``select``.  All ``CK`` key
    columns are written at once (``(in_key, out_key)`` plus ``historical_marker`` when the
    conn gene declares it), so the key and attribute layouts stay aligned.

    Args:
        conn_keys: ``(P, E, CK) int32`` connection key rows, ``I_INF`` padded.
        conn_attrs: ``(P, E, CA) float32`` connection attribute rows, NaN padded.
        rows: ``(P,) int32`` destination row per individual, ``I_INF`` to mean "no write".
        new_keys: ``(P, CK) int32`` caller-allocated, population-unique connection keys.
        new_attrs: ``(P, CA) float32`` attribute rows of the new connections.

    Returns:
        ``(conn_keys, conn_attrs)`` with the new rows written at ``rows``.

    Raises:
        evoxc.ShapeError / evoxc.BackendError: Propagated unchanged (row out of range,
            dtype mismatch).
    """
    return (
        _write_rows(conn_keys, rows, new_keys),
        _write_rows(conn_attrs, rows, new_attrs),
    )


def delete_conn_by_pos(
    conn_keys: evoxc.SymbolicTensor,
    conn_attrs: evoxc.SymbolicTensor,
    rows: evoxc.SymbolicTensor,
) -> tuple[evoxc.SymbolicTensor, evoxc.SymbolicTensor]:
    """Delete connection rows by row position, restoring the padding sentinels.

    Args:
        conn_keys: ``(P, E, CK) int32`` connection key rows.
        conn_attrs: ``(P, E, CA) float32`` connection attribute rows.
        rows: ``(P,) int32`` row to clear per individual, ``I_INF`` to mean "no delete".

    Returns:
        ``(conn_keys, conn_attrs)`` with the ``CK`` key columns of row ``rows`` set to
        ``I_INF`` and its ``CA`` attribute columns set to NaN.

    Raises:
        evoxc.ShapeError / evoxc.BackendError: Propagated unchanged (row out of range).
    """
    pop_size = conn_keys.shape[0]
    key_width, attr_width = conn_keys.shape[2], conn_attrs.shape[2]
    return (
        _write_rows(conn_keys, rows, enp.full((pop_size, key_width), I_INF, dtype=conn_keys.dtype)),
        _write_rows(
            conn_attrs, rows, enp.full((pop_size, attr_width), float("nan"), dtype=conn_attrs.dtype)
        ),
    )


def delete_conns_by_key(
    conn_keys: evoxc.SymbolicTensor,
    conn_attrs: evoxc.SymbolicTensor,
    keys: evoxc.SymbolicTensor,
) -> tuple[evoxc.SymbolicTensor, evoxc.SymbolicTensor]:
    """Clear every connection touching one node key, for any number of rows per individual.

    This is the companion of :func:`delete_conn_by_pos` for node deletion, where the number
    of connections touching the deleted node is data-dependent: it is one elementwise
    ``select`` over a key-match mask combined with a row-validity mask, computed on the whole
    ``(P, E, ...)`` table at once — never a loop and never a per-connection delete.  A genome
    with no connection touching ``keys`` is returned unchanged.

    Args:
        conn_keys: ``(P, E, CK) int32`` connection key rows; columns 0/1 are the source and
            destination node keys, ``I_INF`` marks an absent row.
        conn_attrs: ``(P, E, CA) float32`` connection attribute rows, NaN for an absent row.
        keys: ``(P,) int32`` node key to detach per individual, ``I_INF`` for an individual
            with nothing to delete.

    Returns:
        ``(conn_keys, conn_attrs)`` with every row whose source or destination key equals
        ``keys`` cleared: all ``CK`` key columns set to ``I_INF``, all ``CA`` attribute
        columns set to NaN; all other rows, including absent ones, are untouched.

    Raises:
        evoxc.ShapeError / evoxc.BackendError: Propagated unchanged.

    Notes:
        The row-validity conjunct is not redundant: absent rows carry ``I_INF`` in column 0,
        so an ``I_INF`` "nothing to delete" key would otherwise match every padding row and
        rewrite it — indistinguishable from a real delete in the key array, and a NaN
        overwrite in the attribute array.
    """
    pop_size, edge_count = conn_keys.shape[0], conn_keys.shape[1]
    target = enp.reshape(keys, (pop_size, 1))
    touches = evoxc.logical_or(
        evoxc.equal(conn_keys[:, :, 0], target),
        evoxc.equal(conn_keys[:, :, 1], target),
    )
    drop = evoxc.logical_and(touches, evoxc.not_equal(conn_keys[:, :, 0], _i32(I_INF)))
    # A (P, E, 1) predicate broadcasts over the row width; `select` itself does not
    # broadcast operand ranks, so the mask is expanded explicitly.
    expanded = enp.reshape(drop, (pop_size, edge_count, 1))
    return (
        evoxc.select(expanded, enp.full((), I_INF, dtype=conn_keys.dtype), conn_keys),
        evoxc.select(expanded, enp.full((), float("nan"), dtype=conn_attrs.dtype), conn_attrs),
    )


# --------------------------------------------------------------------------- #
# cycle predicate
# --------------------------------------------------------------------------- #


def check_cycles(
    node_keys: evoxc.SymbolicTensor,
    conn_keys: evoxc.SymbolicTensor,
    src_keys: evoxc.SymbolicTensor,
    dst_keys: evoxc.SymbolicTensor,
) -> evoxc.SymbolicTensor:
    """Ask whether adding one edge per individual would close a cycle.

    The predicate is a *bounded* reachability relaxation over the existing connection set —
    a static Python ``for`` loop over at most ``max_nodes`` hops (never ``while_loop``),
    which is enough because a simple path in a graph of ``N`` nodes has at most ``N-1``
    edges.  A cycle is closed exactly when ``src`` is reachable from ``dst`` in the graph
    that already exists (the new edge ``src -> dst`` closes that path), so a self-loop on an
    existing node answers ``True`` — the same answer
    :func:`tensorneat.common.host_graph.check_cycles` gives.

    Args:
        node_keys: ``(P, N) int32`` node key rows, ``I_INF`` padded.
        conn_keys: ``(P, E, CK) int32`` connection key rows — columns 0/1 are the source and
            destination node keys; absent rows are ``I_INF``.
        src_keys: ``(P,) int32`` new edge source node key, arbitrary (possibly absent) key.
        dst_keys: ``(P,) int32`` new edge destination node key.

    Returns:
        ``(P,) bool`` True where the new edge would close a cycle.  Absent or degenerate
        keys (``src == dst`` with an absent node, either endpoint ``I_INF`` or not a node of
        the genome) return False: the mutation is refused for another reason, not this one.

    Raises:
        evoxc.ShapeError / evoxc.BackendError: Propagated unchanged.

    Notes:
        Cost is split so that no ``max_nodes**2`` adjacency matrix is built: resolving the
        ``E`` edge endpoints to node rows is one ``O(E * N)`` key match (keys are arbitrary,
        so *some* match against the row table is unavoidable), and each hop costs a
        ``O(E)`` masked gather of the source-side reachability plus a ``O(E)`` masked
        scatter back onto the destination rows — the scatter is masked so a dropped write
        cannot clear an already-reachable row (``scatter`` is set-only, so a plain write of
        ``False`` would).  A dense adjacency would cost ``O(N**2)`` per hop on top of the
        same match, and would have to be materialised from the keys anyway.
    """
    pop_size, node_count = node_keys.shape
    edge_count = conn_keys.shape[1]
    node_present = evoxc.not_equal(node_keys, _i32(I_INF))

    src_row, src_hit = _node_rows(src_keys, node_keys, node_present)
    dst_row, dst_hit = _node_rows(dst_keys, node_keys, node_present)
    edge_src, edge_src_hit = _node_rows(conn_keys[:, :, 0], node_keys, node_present)
    edge_dst, edge_dst_hit = _node_rows(conn_keys[:, :, 1], node_keys, node_present)
    edge_valid = evoxc.logical_and(edge_src_hit, edge_dst_hit)

    # Seed the frontier at the destination and grow it one hop at a time; the trailing
    # column is the dump slot of `masked_flat_scatter`.
    dump = enp.zeros((pop_size, 1), dtype=evoxc.bool_)
    seed = masked_flat_scatter(
        enp.zeros((pop_size, node_count + 1), dtype=evoxc.bool_),
        enp.reshape(dst_row, (pop_size, 1)),
        enp.full((pop_size, 1), True, dtype=evoxc.bool_),
        enp.reshape(dst_hit, (pop_size, 1)),
    )
    reachable = evoxc.slice(seed, (0, 0), (pop_size, node_count))
    live_updates = enp.full((pop_size, edge_count), True, dtype=evoxc.bool_)
    for _ in range(node_count - 1):
        live = masked_flat_gather(reachable, edge_src, mask=edge_valid, fill=False)
        padded = evoxc.concatenate((reachable, dump), axis=1)
        reachable = evoxc.slice(
            masked_flat_scatter(padded, edge_dst, live_updates, live),
            (0, 0),
            (pop_size, node_count),
        )

    reached = enp.reshape(
        masked_flat_gather(
            reachable,
            enp.reshape(src_row, (pop_size, 1)),
            mask=enp.reshape(src_hit, (pop_size, 1)),
            fill=False,
        ),
        (pop_size,),
    )
    return evoxc.logical_and(evoxc.logical_and(reached, src_hit), dst_hit)


# --------------------------------------------------------------------------- #
# host-only renumbering
# --------------------------------------------------------------------------- #


def re_cound_idx(
    node_keys: evoxc.SymbolicTensor,
    node_attrs: evoxc.SymbolicTensor,
    conn_keys: evoxc.SymbolicTensor,
    input_idx: list[int],
    output_idx: list[int],
) -> tuple[evoxc.SymbolicTensor, evoxc.SymbolicTensor, evoxc.SymbolicTensor]:
    """Renumber hidden node keys to be contiguous above the input/output keys (host-only).

    Host-only display helper: it converts its arguments with ``Tensor.numpy()``, applies the
    renumbering in Python and returns plain ``numpy`` arrays, which must never be fed back
    into a graph.  Input and output keys are left untouched (they are the external
    interface); hidden keys are replaced by consecutive integers starting at
    ``max(*input_idx, *output_idx) + 1``, in row order, and every connection key referring
    to a hidden node is rewritten accordingly.

    Args:
        node_keys: ``(N,) int32`` node key row of one genome, ``I_INF`` padded.
        node_attrs: ``(N, NA) float32`` node attribute row of the same genome, NaN padded.
        conn_keys: ``(E, CK) int32`` connection key rows of the same genome, ``I_INF``
            padded.
        input_idx: Keys (not rows) of the input nodes, kept verbatim.
        output_idx: Keys of the output nodes, kept verbatim.

    Returns:
        ``(node_keys, node_attrs, conn_keys)`` as ``numpy`` arrays: renumbered node keys,
        the unchanged node attributes (carried along so a payload can be rebuilt in one
        step) and renumbered connection keys.

    Raises:
        ConfigError: If ``input_idx`` and ``output_idx`` are both empty — there is then no
            external interface to renumber the hidden keys above — or if the arrays are
            batched (``node_keys`` not ``(N,)`` / ``conn_keys`` not ``(E, CK)``): a batch
            would otherwise be renumbered row by row through the wrong axis.

    Notes:
        Only columns 0 and 1 of ``conn_keys`` are rewritten: a third column is the
        ``historical_marker``, an identity, not a node reference.
    """
    keys = to_numpy(node_keys)
    attrs = to_numpy(node_attrs)
    conns = to_numpy(conn_keys)
    if keys.ndim != 1 or conns.ndim != 2:
        raise ConfigError(
            "re_cound_idx takes one genome: node_keys (N,) and conn_keys (E, CK), got "
            f"{keys.shape} and {conns.shape}"
        )
    fixed = [int(key) for key in (*input_idx, *output_idx)]
    if not fixed:
        raise ConfigError(
            "re_cound_idx needs at least one input or output key to renumber hidden keys above"
        )
    fixed_set = set(fixed)
    next_key = max(fixed) + 1
    old2new: dict[int, int] = {}
    new_keys = np.array(keys, copy=True)
    for row in range(new_keys.shape[0]):
        key = int(new_keys[row])
        if key == I_INF or key in fixed_set:  # padding row or external interface
            continue
        old2new[key] = next_key
        new_keys[row] = next_key
        next_key += 1
    new_conns = np.array(conns, copy=True)
    for row in range(new_conns.shape[0]):
        for column in (0, 1):  # (in_key, out_key); a marker column is not a reference
            key = int(new_conns[row, column])
            if key in old2new:
                new_conns[row, column] = old2new[key]
    return new_keys, attrs, new_conns
