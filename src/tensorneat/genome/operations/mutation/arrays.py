"""In-graph row plumbing for the structural mutation plan.

Why this module exists (and why it does not just call :mod:`tensorneat.common.tools`):
the structural stage writes whole rows of the four payload arrays and reads a handful of
attribute rows, and the general-purpose helpers there rebuild a flat index block *and* pay
a range-escape check on every call.  The plan computes every row it touches itself, so each
index is in range by construction, and one index block is shared by every read/write of the
same array — which is what keeps this stage's traced-op count proportional to the number of
*distinct* accesses rather than to the number of helper calls.

Everything here is called inside an active trace, takes batch-first tensors and returns
batch-first tensors of static shape; nothing branches on a traced value.
"""

from typing import Any

import numpy as np
import evoxc
import evoxc.numpy as enp

__all__ = [
    "column_base",
    "first_row",
    "flatten",
    "graph_constant",
    "i32",
    "masked_random_row",
    "random_row",
    "row_offsets",
    "take_block_at",
    "take_rows",
    "uniform_bounds",
    "write_rows",
]


def graph_constant(values: Any, dtype: Any = np.int32) -> evoxc.SymbolicTensor:
    """Embed a host array as an in-graph constant.

    ``evoxc`` only accepts a concrete ``Tensor`` there, and a closure-captured host array
    is a ``TraceError``; this is the sanctioned wrapper.  The data is static configuration
    (row indices, column indices, thresholds), never per-generation payload.

    Args:
        values: Host data, any shape.
        dtype: NumPy dtype of the embedded constant.

    Returns:
        ``SymbolicTensor`` with ``values``' shape and ``dtype``.
    """
    return evoxc.constant(evoxc.from_numpy(np.ascontiguousarray(np.asarray(values, dtype=dtype))))


def i32(value: int) -> evoxc.SymbolicTensor:
    """Rank-0 ``int32`` constant holding ``value``.

    A bare Python literal would promote an ``int32`` operand to ``int64`` (plain
    ``np.result_type`` promotion), silently widening every downstream index block.

    Args:
        value: Static integer.

    Returns:
        Rank-0 ``int32`` tensor inside the active trace.
    """
    return enp.full((), int(value), dtype=evoxc.int32)


def uniform_bounds() -> tuple[evoxc.SymbolicTensor, evoxc.SymbolicTensor]:
    """Rank-0 ``float32`` ``0.0`` / ``1.0`` pair to hand to ``evoxc.random.uniform``.

    Passing Python floats as the bounds makes ``evoxc`` embed one ``float64``
    ``constant`` per bound *per draw*, and the reference interpreter charges a
    full op for each of them.  ``evoxc.random.uniform`` also accepts symbolic
    bounds, and the numpy kernel computes ``(low + u * (high - low)).astype(dtype)``
    with the fraction ``u`` in ``float64``; ``0.0`` and ``1.0`` are exact in
    ``float32`` as well, so every drawn value is bit-identical.  Building the
    pair once per stage and sharing it across the stage's draws therefore turns
    the per-draw bound constants into two constants for the whole stage.

    Returns:
        ``(low, high)``: rank-0 ``float32`` tensors holding ``0.0`` and ``1.0``.
    """
    return (
        enp.full((), 0.0, dtype=evoxc.float32),
        enp.full((), 1.0, dtype=evoxc.float32),
    )


def row_offsets(pop_size: int, width: int) -> evoxc.SymbolicTensor:
    """Batch strides ``[0, W, 2W, ...]`` of a flattened ``(P, W)`` array, as ``(P, 1)``.

    Args:
        pop_size: Population size ``P`` (static).
        width: Row width ``W`` of the flattened array (static).

    Returns:
        ``(P, 1)`` ``int32`` tensor that broadcasts against any ``(P, k)`` column block, so
        ``row * width + column + offsets`` is the flat index of the cell.
    """
    return graph_constant(
        np.arange(0, pop_size * width, width, dtype=np.int32).reshape(pop_size, 1)
    )


def column_base(pop_size: int, row_count: int, width: int) -> np.ndarray:
    """Flat offset of every cell of a ``(P, row_count, width)`` block, as ``(P, 1, width)``.

    ``base[p, 0, c] = p * row_count * width + c`` is the batch stride of individual ``p``
    plus the column index, so one row of that block starts at ``base + row * width`` and a
    gather with that index returns the ``(P, 1, width)`` block directly.

    Args:
        pop_size: Population size ``P`` (static).
        row_count: Number of rows per individual ``R`` (static).
        width: Row width ``W`` of the flattened array (static).

    Returns:
        ``(P, 1, W) int32`` host array for :func:`graph_constant`.
    """
    rows = np.arange(pop_size, dtype=np.int32).reshape(pop_size, 1, 1) * np.int32(
        row_count * width
    )
    return (rows + np.arange(width, dtype=np.int32).reshape(1, 1, width)).astype(
        np.int32
    )


def flatten(x: evoxc.SymbolicTensor) -> evoxc.SymbolicTensor:
    """Flatten every axis of ``x`` into one ``1-D`` tensor (one traced op).

    Args:
        x: Any tensor.

    Returns:
        ``(size,)`` tensor with the same dtype; ``gather``/``scatter`` on axis ``0`` then
        address cells by flat index.
    """
    size = 1
    for dim in x.shape:
        size *= int(dim)
    return enp.reshape(x, (size,))


def first_row(
    mask: evoxc.SymbolicTensor,
) -> tuple[evoxc.SymbolicTensor, evoxc.SymbolicTensor]:
    """First ``True`` entry of each batch row, together with the "any entry" flag.

    The pair is returned instead of the ``I_INF`` sentinel that
    :func:`tensorneat.common.fetch_first` writes into the index, because the caller's gate
    is a boolean ``and`` chain anyway: a ``select`` plus a rank-0 sentinel constant costs two
    traced ops per call, and three of them live in the node-split plan alone.  The index of a
    row with no ``True`` entry is ``0`` — always in range, and never observable, because the
    flag is false for that individual.

    Args:
        mask: ``(P, L) bool`` eligibility mask.

    Returns:
        ``(row, hit)``: ``(P,) int32`` first-``True`` column (``0`` where the row is empty)
        and ``(P,) bool`` "this row has a ``True`` entry".
    """
    hit = evoxc.reduce_max(mask, axes=1)
    row = evoxc.cast(evoxc.argmax(mask, axis=1), evoxc.int32)
    return row, hit


def random_row(
    key: evoxc.SymbolicTensor,
    mask: evoxc.SymbolicTensor,
    low: "evoxc.SymbolicTensor | None" = None,
    high: "evoxc.SymbolicTensor | None" = None,
) -> tuple[evoxc.SymbolicTensor, evoxc.SymbolicTensor]:
    """Uniformly random ``True`` entry of each batch row, with its "any entry" flag.

    The maximiser of i.i.d. uniforms restricted to the ``True`` set is uniform over that set,
    so masking the draws with a ``multiply`` (one op, no ``-inf`` constant and no ``select``)
    and taking the ``argmax`` samples one eligible column per individual.  Draw order is
    therefore *not* the legacy ``cumsum``-plus-rank order — that is already true of every
    draw in this node.

    Args:
        key: Rank-0 ``evoxc.random`` key; the whole batch comes from this one key.
        mask: ``(P, L) bool`` eligibility mask.
        low: Optional rank-0 ``float32`` lower bound; ``0.0`` when omitted.
        high: Optional rank-0 ``float32`` upper bound; ``1.0`` when omitted.

    Returns:
        ``(row, hit)``: ``(P,) int32`` chosen column (``0`` where the row is empty) and
        ``(P,) bool`` "this row has a ``True`` entry".
    """
    hit = evoxc.reduce_max(mask, axes=1)
    return masked_random_row(key, evoxc.cast(mask, evoxc.float32), hit, low, high)


def masked_random_row(
    key: evoxc.SymbolicTensor,
    weights: evoxc.SymbolicTensor,
    hit: evoxc.SymbolicTensor,
    low: "evoxc.SymbolicTensor | None" = None,
    high: "evoxc.SymbolicTensor | None" = None,
) -> tuple[evoxc.SymbolicTensor, evoxc.SymbolicTensor]:
    """Draw one sampled column out of an already masked weight block.

    The float copy of an eligibility mask and its "any entry" flag are both
    mask-only values: when two operations draw against the *same* mask (add-node
    and delete-connection both sample the present connections) the stage builds
    them once and passes them in, so this helper is the bare draw, its
    restriction and the ``argmax`` — the only per-call work left.

    Args:
        key: Rank-0 ``evoxc.random`` key; the whole batch comes from this one key.
        weights: ``(P, L) float32`` eligibility mask, ``1.0`` where eligible and
            ``0.0`` elsewhere.
        hit: ``(P,) bool`` "this row has an eligible entry" (``any(mask, axis=1)``).
        low: Optional rank-0 ``float32`` lower bound (``0.0`` when omitted); a
            shared tensor instead of a per-call float ``constant`` (see
            :func:`uniform_bounds`), which draws bit-identically.
        high: Optional rank-0 ``float32`` upper bound (``1.0`` when omitted).

    Returns:
        ``(row, hit)``: ``(P,) int32`` chosen column (``0`` where the row is empty) and the
        ``hit`` flag passed in.
    """
    draws = evoxc.random.uniform(
        key,
        tuple(weights.shape),
        0.0 if low is None else low,
        1.0 if high is None else high,
        dtype=evoxc.float32,
    )
    row = evoxc.cast(evoxc.argmax(evoxc.multiply(draws, weights), axis=1), evoxc.int32)
    return row, hit


def take_rows(
    x_flat: evoxc.SymbolicTensor,
    offsets: evoxc.SymbolicTensor,
    rows: evoxc.SymbolicTensor,
) -> evoxc.SymbolicTensor:
    """Gather one cell per ``(P, k)`` (or ``(P,)``) row index out of a flattened payload.

    Args:
        x_flat: ``(P * R,)`` flattened batch-first source, one row of ``R`` cells per
            individual.
        offsets: ``(P, 1)`` batch strides from :func:`row_offsets` (``R`` as the width).
        rows: ``(P,)`` or ``(P, k)`` ``int32`` row indices in ``[0, R)``.

    Returns:
        ``rows.shape`` tensor of ``x``'s dtype.  An index outside ``[0, R)`` raises in the
        backend — this helper does not sanitize, because its callers own the domains.

    Raises:
        evoxc.ShapeError / evoxc.BackendError: Propagated unchanged.
    """
    flat_rows = enp.reshape(rows, (rows.shape[0], -1))
    index = enp.reshape(evoxc.add(flat_rows, offsets), (-1,))
    return enp.reshape(evoxc.gather(x_flat, index, axis=0), tuple(rows.shape))


def take_block_at(
    x_flat: evoxc.SymbolicTensor,
    base: evoxc.SymbolicTensor,
    rows_column: evoxc.SymbolicTensor,
    width: evoxc.SymbolicTensor,
) -> evoxc.SymbolicTensor:
    """Gather the ``width`` cells of one row per individual, out of a flattened payload.

    The flat index of every cell is ``base + row * width``: ``base`` carries the
    batch stride *and* the column index of each cell, so the block comes out in
    one broadcast ``add`` and the gather returns the ``(P, 1, width)`` block
    directly — no index reshape and no output reshape.  Two reads of the same row
    (connection keys and connection attributes) share ``rows_column`` and each
    costs three traced ops.

    Args:
        x_flat: ``(P * R * width,)`` flattened batch-first source.
        base: ``(P, 1, width) int32`` constant ``p * R * width + column``.
        rows_column: ``(P, 1, 1) int32`` row indices in ``[0, R)`` (rank 3 so the
            broadcast pairs the row axis, not the population axis).
        width: Rank-0 ``int32`` row width ``width``.

    Returns:
        ``(P, 1, width)`` tensor of ``x``'s dtype.  An index outside ``[0, R)`` raises in
        the backend — this helper does not sanitize, because its callers own the domains.

    Raises:
        evoxc.ShapeError / evoxc.BackendError: Propagated unchanged.
    """
    return evoxc.gather(
        x_flat, evoxc.add(base, evoxc.multiply(rows_column, width)), axis=0
    )


def write_rows(
    x: evoxc.SymbolicTensor,
    row_index: evoxc.SymbolicTensor,
    rows: evoxc.SymbolicTensor,
    values: evoxc.SymbolicTensor,
) -> evoxc.SymbolicTensor:
    """Overwrite one row per individual with a ``select`` on a row-index comparison.

    A row index equal to ``x``'s own row count matches no row, which is exactly
    the "no write" slot the plan uses; no dump row, no flattening and no
    out-of-range index is needed, and an individual whose plan writes nothing
    keeps its row bit-for-bit (``-0.0`` and ``NaN`` included).

    Args:
        x: ``(P, R)`` or ``(P, R, W)`` payload to write into.
        row_index: ``(1, R)`` (rank 2) or ``(1, R, 1)`` (rank 3) ``int32`` constant
            ``[0, 1, ..., R-1]`` — the rank sets the rank of the written values.
        rows: ``(P,) int32`` destination rows, ``R`` meaning "no write".
        values: Written value, broadcastable to ``x``'s shape (``(P, 1)`` for a rank-2
            payload, ``(P, 1, W)`` for a rank-3 one, or a rank-0 constant).

    Returns:
        ``x`` with ``rows`` overwritten, same shape and dtype.

    Raises:
        evoxc.ShapeError / evoxc.BackendError: Propagated unchanged.
    """
    mask = evoxc.equal(
        row_index,
        enp.reshape(rows, (rows.shape[0],) + (1,) * (len(x.shape) - 1)),
    )
    return evoxc.select(mask, values, x)
