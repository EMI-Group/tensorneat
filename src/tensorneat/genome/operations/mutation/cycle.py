"""Cycle detection for one candidate edge per individual, in a single loop region.

The predicate answers "would adding ``src -> dst`` close a cycle?", i.e. "is ``src`` already
reachable from ``dst``?".  It replaces the unrolled ``max_nodes - 1`` hop relaxation of
:func:`tensorneat.genome.utils.check_cycles` with **one** ``evoxc.while_loop``: the reference
interpreter pays per traced op, so 49 hand-unrolled hops (≈360 traced ops, each executed) cost
far more than a 7-op loop body that runs once per *real* hop of the shallow DAGs a NEAT run
produces.

The loop is a monotone set-growth relaxation on a **flat** frontier with one trailing dump
column (``scatter`` is set-only, so a refused edge writes the dump instead of clearing a real
node).  Growth is bounded by the node count, so termination is guaranteed by the set becoming
stable — the static ``step < max_nodes`` term is a second, cheap cap.
"""

import numpy as np
import evoxc
import evoxc.numpy as enp

from ....common import I_INF


def _static_constant(values: np.ndarray) -> evoxc.SymbolicTensor:
    """Embed a host-side ``int32`` array as a single graph constant.

    Args:
        values (np.ndarray): An ``int32`` array of the shape the kernel needs.

    Returns:
        evoxc.SymbolicTensor: The same values inside the graph, one traced op.
    """
    return evoxc.constant(evoxc.from_numpy(np.asarray(values, dtype=np.int32)))

__all__ = ["closes_cycle"]


def _row_of(
    queries: evoxc.SymbolicTensor,
    node_keys: evoxc.SymbolicTensor,
    node_present: evoxc.SymbolicTensor,
) -> tuple[evoxc.SymbolicTensor, evoxc.SymbolicTensor]:
    """Resolve each queried node key to the row that carries it.

    Node keys are arbitrary and a node may sit on any row, so a key can only be resolved by
    matching it against the row table — one ``O(N)``-wide pass for the whole query block,
    never one pass per key.  All the queries of :func:`closes_cycle` (both endpoints of every
    existing edge plus both endpoints of the candidate edge) share this single pass.

    Args:
        queries: ``(P, Q) int32`` node keys to look up.
        node_keys: ``(P, N) int32`` node key per row, ``I_INF`` padded.
        node_present: ``(P, N) bool`` row-validity mask (``node_keys != I_INF``).

    Returns:
        ``(rows, hit)``, both ``(P, Q)``: the row index of the matching node (``argmax``'s
        fallback row ``0`` where nothing matched) and the match mask.  Every consumer must
        gate on ``hit`` — a fallback row is a real row.
    """
    pop_size, node_count = node_keys.shape
    query_count = queries.shape[1]
    match = evoxc.logical_and(
        evoxc.equal(
            enp.reshape(queries, (pop_size, query_count, 1)),
            enp.reshape(node_keys, (pop_size, 1, node_count)),
        ),
        enp.reshape(node_present, (pop_size, 1, node_count)),
    )
    rows = evoxc.cast(evoxc.argmax(evoxc.cast(match, evoxc.int32), axis=-1), evoxc.int32)
    return rows, evoxc.reduce_max(match, axes=-1)


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
        One ``while_loop`` carries the flat frontier, the iteration counter and the
        "still growing" flag; its body is four ``O(P*E)`` ops (gather, ``and``, ``select``,
        ``scatter``) plus the two-op growth test, and it runs once per real hop.  The
        batch axis is written explicitly — no ``vmap`` over the region.
    """
    pop_size, node_count = node_keys.shape
    edge_count = conn_keys.shape[1]
    width = node_count + 1
    node_present = evoxc.not_equal(node_keys, I_INF)
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
    rows, hits = _row_of(queries, node_keys, node_present)
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
    dump_flat = _static_constant(
        np.repeat(
            np.arange(0, pop_size * width, width, dtype=np.int32) + node_count,
            edge_count,
        )
    )
    valid_flat = enp.reshape(edge_valid, (edge_positions,))
    # Seed the frontier at the destination: a one-hot row, then a zero dump column.
    seeded = evoxc.logical_and(
        evoxc.equal(
            enp.arange(0, node_count, dtype=evoxc.int32), dst_row
        ),
        dst_hit,
    )
    frontier = enp.reshape(
        enp.concatenate(
            [seeded, enp.zeros((pop_size, 1), dtype=evoxc.bool_)], axis=1
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
