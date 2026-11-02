"""Cycle detection for one candidate edge per individual, in a single loop region.

The predicate answers "would adding ``src -> dst`` close a cycle?", i.e. "is ``src`` already
reachable from ``dst``?".  It replaces the unrolled ``max_nodes - 1`` hop relaxation of
:func:`tensorneat.genome.utils.check_cycles` with **one** ``evoxc.while_loop``: the reference
interpreter pays per traced op, so 49 hand-unrolled hops (≈360 traced ops, each executed) cost
far more than a 7-op loop body that runs once per *real* hop of the shallow DAGs a NEAT run
produces.

Key resolution is a **merge join**, not a broadcast match.
Node keys are arbitrary ``int32`` values — the mutation operator allocates them as
``max_node_key + 1 + arange(pop_size)``, so they routinely leave ``genome.key_span`` and grow
without bound, which rules out any dense ``(P, key_span)`` scatter table.
The old resolve compared the whole query block against the whole row table, an
``(P, 2E + 2, N)`` boolean (10.1 M elements at pop 1000) that it then cast to ``int32`` and
``argmax``-ed; that cost 21 of the predicate's 23 ms.
The join below instead folds every key together with its payload into one injective ``int64``
composite, sorts the ``(P, N + Q)`` merge once, and reads each query's node off the sorted
order (``N + Q = 252`` elements per individual, a 40× volume cut).
The low field keeps queries and nodes apart — queries occupy ``[0, Q)``, nodes ``[Q, Q + N)``
in row order — so within a key run the queries sort first and the nodes follow in ascending row
order: the first node at or after a query position is the *smallest row* that carries the key,
which is exactly what ``argmax`` of the old match returned.

The loop is a monotone set-growth relaxation on a **flat** frontier with one trailing dump
column (``scatter`` is set-only, so a refused edge writes the dump instead of clearing a real
node).  Growth is bounded by the node count, so termination is guaranteed by the set becoming
stable — the static ``step < max_nodes`` term is a second, cheap cap.
"""

import numpy as np
import evoxc
import evoxc.numpy as enp

from ....common import I_INF
from ...window import fill


def _static_constant(values: np.ndarray) -> evoxc.SymbolicTensor:
    """Embed a host-side ``int32`` array as a single graph constant.

    Args:
        values (np.ndarray): An ``int32`` array of the shape the kernel needs.

    Returns:
        evoxc.SymbolicTensor: The same values inside the graph, one traced op.
    """
    return evoxc.constant(evoxc.from_numpy(np.asarray(values, dtype=np.int32)))


def _static_wide(values: np.ndarray) -> evoxc.SymbolicTensor:
    """Embed a host-side ``int64`` array as a single graph constant.

    Args:
        values (np.ndarray): An ``int64`` array of the shape the kernel needs.

    Returns:
        evoxc.SymbolicTensor: The same values inside the graph, one traced op.
    """
    return evoxc.constant(evoxc.from_numpy(np.asarray(values, dtype=np.int64)))


__all__ = ["closes_cycle"]


def _row_of(
    queries: evoxc.SymbolicTensor,
    node_keys: evoxc.SymbolicTensor,
) -> tuple[evoxc.SymbolicTensor, evoxc.SymbolicTensor]:
    """Resolve each queried node key to the row that carries it, by one sorted merge.

    Node keys are arbitrary and a node may sit on any row, so a key can only be resolved by
    comparing it against the row table.  Doing that comparison as a broadcast produces an
    ``(P, Q, N)`` block; this join instead merges the two sorted key lists, so it touches
    ``P * (N + Q)`` elements instead of ``P * Q * N`` and needs no ``argmax``.

    Args:
        queries: ``(P, Q) int32`` node keys to look up.
        node_keys: ``(P, N) int32`` node key per row, ``I_INF`` padded.

    Returns:
        ``(rows, hit)``, both ``(P, Q)``: the row of the *smallest* matching node, and the
        match mask.  ``rows`` is in ``[0, N)`` even where ``hit`` is false (those slots are
        unconstrained), so it is always safe to index with; every consumer must still gate on
        ``hit``, because an unmatched key resolves to a real, unrelated row.

    Notes:
        A padding node key is nudged past ``I_INF`` inside the composite, so an ``I_INF`` query
        — an absent edge endpoint — can never match a padding row and the caller needs no
        separate padding mask.  The ``int64`` composite is ``key << bits | low`` with
        ``2 ** (bits - 1) > N + Q - 1``, so two entries of the same key lie at most
        ``N + Q - 1`` apart in the low field while two entries of different keys lie at least
        a whole period apart: the hit test is one range test on the difference.
    """
    pop_size, node_count = node_keys.shape
    query_count = queries.shape[1]
    total = node_count + query_count
    bits = (total - 1).bit_length() + 1
    low_base = 1 << bits
    c_shift = _static_wide(np.asarray(bits))
    c_radius = _static_wide(np.asarray(low_base >> 1))
    c_mask = _static_constant(np.asarray(low_base - 1))
    c_query = _static_constant(np.asarray(query_count))
    # A padding row is nudged to I_INF + 1, a key no query can hold, so it never matches.
    bumped = evoxc.add(
        evoxc.cast(node_keys, evoxc.int64),
        evoxc.cast(evoxc.equal(node_keys, _static_constant(np.asarray(I_INF))), evoxc.int64),
    )
    node_enc = evoxc.add(
        evoxc.bitwise_left_shift(bumped, c_shift),
        _static_wide((query_count + np.arange(node_count)).astype(np.int64)),
    )
    query_enc = evoxc.add(
        evoxc.bitwise_left_shift(evoxc.cast(queries, evoxc.int64), c_shift),
        _static_wide(np.arange(query_count).astype(np.int64)),
    )
    ordered = evoxc.sort(enp.concatenate([node_enc, query_enc], axis=1), axis=1)
    # ``cast`` to int32 truncates, which already hands back the low field; the mask clears the
    # key bits that survive above it.
    low = evoxc.bitwise_and(evoxc.cast(ordered, evoxc.int32), c_mask)
    remaining = evoxc.cumsum(
        evoxc.cast(evoxc.greater_equal(low, c_query), evoxc.int32), axis=1, reverse=True
    )
    # ``remaining`` counts the nodes at or after the position, so ``node_count - remaining`` is
    # the rank of the first one; a trailing sentinel column keeps that index in range when no
    # node follows (its key is I_INF + 1, so it never passes the hit test).
    rank = evoxc.subtract(_static_constant(np.asarray(node_count)), remaining)
    # One sentinel *element*, broadcast down the column: the value is static data with no
    # per-individual content, so a materialised ``(P, 1)`` int64 column would only be a
    # graph constant proportional to ``P``.
    sorted_nodes = enp.concatenate(
        [
            evoxc.sort(node_enc, axis=1),
            enp.broadcast_to(
                _static_wide(np.asarray((I_INF + 1) << bits)), (pop_size, 1)
            ),
        ],
        axis=1,
    )
    stride = node_count + 1
    flat_nodes = enp.reshape(sorted_nodes, (pop_size * stride,))
    offsets = _static_constant(np.arange(0, pop_size * stride, stride).reshape(pop_size, 1))
    pick = enp.reshape(evoxc.add(rank, offsets), (pop_size * total,))
    node_at_rank = enp.reshape(evoxc.gather(flat_nodes, pick, axis=0), (pop_size, total))
    hit = evoxc.less(evoxc.abs(evoxc.subtract(ordered, node_at_rank)), c_radius)
    rows = evoxc.subtract(
        evoxc.bitwise_and(evoxc.cast(node_at_rank, evoxc.int32), c_mask), c_query
    )
    # Scatter the answers back into query order: a hit writes its row, a miss writes -1 into
    # the same cell, and every node position writes the trailing dump column.
    answer = evoxc.select(hit, rows, _static_constant(np.asarray(-1)))
    width = query_count + 1
    store = _static_constant(np.arange(0, pop_size * width, width).reshape(pop_size, 1))
    written = evoxc.scatter(
        enp.reshape(
            fill((pop_size, width), 0, dtype=evoxc.int32), (pop_size * width,)
        ),
        enp.reshape(evoxc.add(evoxc.minimum(low, c_query), store), (pop_size * total,)),
        enp.reshape(answer, (pop_size * total,)),
        axis=0,
    )
    packed = evoxc.slice(
        enp.reshape(written, (pop_size, width)), (0, 0), (pop_size, query_count)
    )
    return (
        evoxc.maximum(packed, _static_constant(np.asarray(0))),
        evoxc.greater_equal(packed, _static_constant(np.asarray(0))),
    )


def closes_cycle(
    node_keys: evoxc.SymbolicTensor,
    conn_keys: evoxc.SymbolicTensor,
    src_keys: evoxc.SymbolicTensor,
    dst_keys: evoxc.SymbolicTensor,
) -> evoxc.SymbolicTensor:
    """Ask whether adding one edge per individual would close a cycle.

    A cycle is closed exactly when ``src`` is reachable from ``dst`` in the graph that
    already exists, so the predicate seeds a frontier at ``dst`` and grows it along the
    existing edges until it stops growing; a self-loop on an existing node answers ``True``.

    Args:
        node_keys: ``(P, N) int32`` node key rows, ``I_INF`` padded.
        conn_keys: ``(P, E, CK) int32`` connection key rows — columns ``0``/``1`` are the
            source and destination node keys; absent rows are ``I_INF``.
        src_keys: ``(P,) int32`` candidate edge source node key.
        dst_keys: ``(P,) int32`` candidate edge destination node key.

    Returns:
        ``(P,) bool`` True where the candidate edge would close a cycle.  A degenerate
        candidate (either endpoint absent from the genome, or ``I_INF``) answers ``False``:
        the mutation is refused for another reason, never this one.

    Raises:
        evoxc.ShapeError / evoxc.BackendError: Propagated unchanged.

    Notes:
        Both endpoints of every existing edge plus both endpoints of the candidate edge are
        resolved in one :func:`_row_of` merge, so the whole predicate pays that join once.
        One ``while_loop`` then carries the flat frontier, the iteration counter and the
        "still growing" flag; its body is four ``O(P*E)`` ops (gather, ``and``, ``select``,
        ``scatter``) plus the two-op growth test, and it runs once per real hop.  The batch
        axis is written explicitly — no ``vmap`` over the region.
    """
    pop_size, node_count = node_keys.shape
    edge_count = conn_keys.shape[1]
    width = node_count + 1
    # Both endpoints of every existing edge plus the candidate edge, in one key-match pass.
    queries = enp.concatenate(
        [
            conn_keys[:, :, 0],
            conn_keys[:, :, 1],
            enp.reshape(src_keys, (pop_size, 1)),
            enp.reshape(dst_keys, (pop_size, 1)),
        ],
        axis=1,
    )
    rows, hits = _row_of(queries, node_keys)
    edge_src = evoxc.slice(rows, (0, 0), (pop_size, edge_count))
    edge_dst = evoxc.slice(rows, (0, edge_count), (pop_size, edge_count))
    src_hit = evoxc.slice(hits, (0, 2 * edge_count), (pop_size, 1))
    dst_hit = evoxc.slice(hits, (0, 2 * edge_count + 1), (pop_size, 1))
    src_row = evoxc.slice(rows, (0, 2 * edge_count), (pop_size, 1))
    dst_row = evoxc.slice(rows, (0, 2 * edge_count + 1), (pop_size, 1))
    edge_valid = evoxc.logical_and(
        evoxc.slice(hits, (0, 0), (pop_size, edge_count)),
        evoxc.slice(hits, (0, edge_count), (pop_size, edge_count)),
    )
    # Flat index blocks, built once: every hop then indexes the frontier directly.  The row
    # offsets and the dump column are pure static data, so they are graph constants (one op
    # each) rather than an ``arange`` plus a broadcast.
    offsets = _static_constant(
        np.arange(0, pop_size * width, width, dtype=np.int32).reshape(pop_size, 1)
    )
    edge_positions = pop_size * edge_count
    src_flat = enp.reshape(evoxc.add(edge_src, offsets), (edge_positions,))
    dst_flat = enp.reshape(evoxc.add(edge_dst, offsets), (edge_positions,))
    # The dump column's flat index is the per-individual row offset plus ``node_count``,
    # identical for every edge of that individual: ``(P, 1)`` static data broadcast across
    # the edge axis instead of a materialised ``(P * E,)`` constant (``P * E`` entries --
    # 40 k, 200 k and 400 k at pop 100/500/1000).
    dump_flat = enp.reshape(
        enp.broadcast_to(
            evoxc.add(offsets, _static_constant(np.asarray(node_count))),
            (pop_size, edge_count),
        ),
        (edge_positions,),
    )
    valid_flat = enp.reshape(edge_valid, (edge_positions,))
    # Seed the frontier at the destination: a one-hot row, then a zero dump column.
    seeded = evoxc.logical_and(
        evoxc.equal(
            enp.reshape(enp.arange(0, node_count, dtype=evoxc.int32), (1, node_count)),
            dst_row,
        ),
        dst_hit,
    )
    frontier = enp.reshape(
        enp.concatenate(
            [seeded, fill((pop_size, 1), False, dtype=evoxc.bool_)], axis=1
        ),
        (pop_size * width,),
    )
    limit = _static_constant(np.asarray(node_count, dtype=np.int32))
    one = _static_constant(np.asarray(1, dtype=np.int32))

    def cond(carried):
        """Continue while the frontier gained a node and the hop cap is not reached."""
        step, _frontier, growing = carried
        return evoxc.logical_and(growing, evoxc.less(step, limit))

    def body(carried):
        """One hop: every edge whose source is reached marks its destination."""
        step, frontier, _growing = carried
        live = evoxc.logical_and(
            evoxc.gather(frontier, src_flat, axis=0), valid_flat
        )
        grown = evoxc.scatter(
            frontier, evoxc.select(live, dst_flat, dump_flat), live, axis=0
        )
        return (
            evoxc.add(step, one),
            grown,
            evoxc.reduce_max(evoxc.not_equal(grown, frontier), axes=0),
        )

    _, frontier, _ = evoxc.while_loop(
        cond,
        body,
        (
            enp.zeros((), dtype=evoxc.int32),
            frontier,
            enp.full((), True, dtype=evoxc.bool_),
        ),
    )
    # Read the frontier at the candidate source's slot: flat index = row offset + column.
    src_slot = enp.reshape(evoxc.add(src_row, offsets), (pop_size,))
    reached = enp.reshape(
        evoxc.gather(frontier, src_slot, axis=0), (pop_size, 1)
    )
    return enp.reshape(
        evoxc.logical_and(evoxc.logical_and(reached, src_hit), dst_hit), (pop_size,)
    )
