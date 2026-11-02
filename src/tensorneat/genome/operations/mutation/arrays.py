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
    "first_row",
    "flatten",
    "graph_constant",
    "i32",
    "random_row",
    "row_offsets",
    "scatter_rows",
    "take_block",
    "take_rows",
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

    Returns:
        ``(row, hit)``: ``(P,) int32`` chosen column (``0`` where the row is empty) and
        ``(P,) bool`` "this row has a ``True`` entry".
    """
    draws = evoxc.random.uniform(
        key, tuple(mask.shape), 0.0, 1.0, dtype=evoxc.float32
    )
    weights = evoxc.multiply(draws, evoxc.cast(mask, evoxc.float32))
    row = evoxc.cast(evoxc.argmax(weights, axis=1), evoxc.int32)
    return row, evoxc.reduce_max(mask, axes=1)


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


def take_block(
    x_flat: evoxc.SymbolicTensor,
    offsets: evoxc.SymbolicTensor,
    rows: evoxc.SymbolicTensor,
    columns: evoxc.SymbolicTensor,
    width: int,
) -> evoxc.SymbolicTensor:
    """Gather the ``width`` cells of every indexed row out of a flattened payload.

    Args:
        x_flat: ``(P * R,)`` flattened batch-first source whose rows are ``width`` wide.
        offsets: ``(P, 1)`` batch strides from :func:`row_offsets` (``R`` as the width).
        rows: ``(P,)`` ``int32`` row indices in ``[0, R / width)``.
        columns: ``(1, width)`` ``int32`` constant ``[0, 1, ..., width - 1]``.
        width: Number of columns per row (static; needed for the flat index arithmetic).

    Returns:
        ``(P, width)`` tensor of ``x``'s dtype.

    Raises:
        evoxc.ShapeError / evoxc.BackendError: Propagated unchanged.
    """
    span = evoxc.add(
        evoxc.multiply(enp.reshape(rows, (rows.shape[0], 1)), i32(width)), columns
    )
    index = enp.reshape(evoxc.add(span, offsets), (-1,))
    return enp.reshape(evoxc.gather(x_flat, index, axis=0), (rows.shape[0], width))


def scatter_rows(
    x: evoxc.SymbolicTensor,
    rows: evoxc.SymbolicTensor,
    values: evoxc.SymbolicTensor,
    dump_row: int,
) -> evoxc.SymbolicTensor:
    """Write one trailing-column-per-individual block of rows into a payload.

    The payload is extended with a trailing **dump row** per individual, so a row index of
    ``dump_row`` absorbs a "no write" update instead of corrupting a real row (a scatter can
    never be a per-individual no-op) and the caller needs no per-array ``select`` guard.
    Rows are written in flat traversal order, so a later slot overwrites an earlier one at
    the same row — the documented duplicate-index contract of ``evoxc.scatter``.

    Args:
        x: ``(P, M, R)`` or ``(P, M)`` payload to write into.
        rows: ``(P, k)`` ``int32`` destination rows in ``[0, M]``, ``M`` meaning the dump
            row, ``k`` the number of row slots per individual.
        values: ``(P, k, R)`` (or ``(P, k)``) values, ``x``'s dtype.
        dump_row: ``M``, the index of the trailing dump row.

    Returns:
        ``(P, M, R)`` / ``(P, M)`` tensor: ``x`` with the ``k`` rows written.

    Raises:
        evoxc.ShapeError / evoxc.BackendError: Propagated unchanged.
    """
    pop_size, row_count = x.shape[0], x.shape[1]
    two_dimensional = len(x.shape) == 2
    width = 1 if two_dimensional else x.shape[2]
    if two_dimensional:
        dump = enp.zeros((pop_size, 1), dtype=x.dtype)
    else:
        dump = enp.zeros((pop_size, 1, width), dtype=x.dtype)
    extended = enp.concatenate([x, dump], axis=1)
    offsets = row_offsets(pop_size, (row_count + 1) * width)
    # A flat index needs the batch stride added per individual; the stride block must carry
    # the same rank as the span it is added to, or the broadcast would pair two axes and
    # multiply the index count by `P`.
    if two_dimensional:
        index = enp.reshape(evoxc.add(rows, offsets), (-1,))
    else:
        columns = graph_constant(np.arange(width, dtype=np.int32).reshape(1, 1, width))
        span = evoxc.add(
            evoxc.multiply(enp.reshape(rows, (pop_size, rows.shape[1], 1)), i32(width)),
            columns,
        )
        index = enp.reshape(
            evoxc.add(span, enp.reshape(offsets, (pop_size, 1, 1))), (-1,)
        )
    written = enp.reshape(
        evoxc.scatter(flatten(extended), index, flatten(values), axis=0),
        (pop_size, row_count + 1, width),
    )
    if two_dimensional:
        written = enp.reshape(written, (pop_size, row_count + 1))
    return evoxc.slice(written, (0, 0, 0)[: len(x.shape)], tuple(x.shape))
