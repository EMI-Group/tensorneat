"""Flat-gather plumbing of the window step: slot indices, column widening, prepared gather.

This module is the *index arithmetic* half of the forward pass — everything
:meth:`tensorneat.genome.base.BaseGenome._hit_conn_attrs` derives once per pass so an unrolled
level pays only for the values themselves. It sits beside ``base.py`` (which keeps the genome
class and the window step) so that file stays one job and under its line budget.

Op count, not element count, is the budget on the numpy backend (one traced op costs about as
much as eighty microseconds at ``P=1000``, whatever its shape), so every helper here is written
to emit the fewest ops that stay bit-identical:

* ``evoxc.gather`` has ``np.take`` semantics and splices the *whole* index shape into the axis
  it indexes, so a batch-wide flat index is the only one-dimensional form that works. The
  per-individual offset ``p * row_len`` is static, hence a single constant (``_row_offsets``)
  instead of an in-graph ``arange``/``multiply``/``reshape`` chain.
* The padding sentinel is a *rank-0* ``I_INF`` (``_iinf``), never a full-size array: a
  ``(P, N, K, CA)`` constant is 3.2 MB at ``P=1000`` and ``evoxc`` warns about embedding it.
* ``select`` and ``logical_and`` broadcast, so the attribute-column mask is widened with a
  trailing-``1`` axis (``_column_mask``) instead of a ``broadcast_to`` op.

No state, no genome class: every function is a pure tensor transformation of its arguments, and
every static (``row_len``, ``attr_width``, shapes) is a Python value fixed at trace time.
"""

from typing import Any

import evoxc
import evoxc.numpy as enp
import numpy as np

from ..common import I_INF, TensorNEATError


def _i32(value: int) -> evoxc.SymbolicTensor:
    """Rank-0 ``int32`` constant.

    Written out instead of a bare Python literal because an ``int32`` tensor plus a Python
    ``int`` promotes to ``int64`` in ``evoxc``, which would silently widen index arithmetic.

    Args:
        value: Static integer baked into the trace.

    Returns:
        Rank-0 ``int32`` tensor inside the active trace.
    """
    return enp.full((), int(value), dtype=evoxc.int32)


def _slot_index(
    idx: evoxc.SymbolicTensor,
    mask: evoxc.SymbolicTensor,
    row_len: int,
) -> tuple[evoxc.SymbolicTensor, evoxc.SymbolicTensor]:
    """Flatten one window index table to a batch-wide gather index plus a liveness mask.
    ``evoxc.gather`` follows ``np.take`` semantics — it splices the *whole* index shape into
    the axis it indexes — so an index that already carries the population axis cannot gather
    along the table's second axis. The batch-wide flat index ``p * row_len + row`` is the
    form that works, and it keeps the gather one-dimensional.
    Both results depend only on the schedule (``idx`` and ``mask``) and on the static
    ``row_len``, so :meth:`BaseGenome._hit_conn_attrs` derives them once per forward pass and
    every unrolled level reuses them instead of re-deriving them.
    Args:
        idx: ``(P, *block) int32`` row indices into the source table's second axis; ``I_INF``
            (or a masked-out position) means "no slot".
        mask: ``(P, *block) bool`` liveness of each slot.
        row_len: ``R``, the source table's second-axis length.
    Returns:
        ``(flat_index, present)``: the ``(P * prod(block),)`` flat index with every absent
        slot redirected to row ``0`` of its own individual (the backend never sees an
        out-of-range index) and the ``(P, *block) bool`` liveness mask — block-shaped, which
        is the shape the per-level value step ANDs against its ``isnan`` test.
    """
    pop_size = int(idx.shape[0])
    block = tuple(int(size) for size in idx.shape[1:])
    count = 1
    for size in block:
        count *= size
    present = evoxc.logical_and(enp.not_equal(idx, _iinf(idx.dtype)), mask)
    safe = evoxc.select(present, idx, enp.zeros((), dtype=idx.dtype))
    flat = evoxc.add(safe, _row_offsets(pop_size, row_len, len(block)))
    return evoxc.reshape(flat, (pop_size * count,)), present


def _row_offsets(pop_size: int, row_len: int, block_rank: int) -> evoxc.SymbolicTensor:
    """``p * row_len`` baked in as one constant, shaped to broadcast over ``block_rank`` axes.

    The per-individual offset of the flat gather is fully static — ``row_len`` is a Python int
    — so it is embedded as a constant instead of being rebuilt in-graph from an ``arange``
    (three fewer traced ops per window index, and one constant instead of two).

    Args:
        pop_size: ``P``, the population size.
        row_len: ``R``, the source table's second-axis length.
        block_rank: Rank of the window block the offsets must broadcast over.

    Returns:
        ``(P, 1, ..., 1)`` ``int32`` constant holding ``p * R``.

    Raises:
        TensorNEATError: If the largest flat index ``(P - 1) * R + R - 1`` does not fit the
            ``int32`` gather index — a silent wrap would read the wrong row.
    """
    largest = pop_size * row_len - 1
    if largest > np.iinfo(np.int32).max:
        raise TensorNEATError(
            f"flat gather index {largest} exceeds int32 (P={pop_size}, R={row_len}); "
            "shrink the population or the table width"
        )
    offsets = np.arange(pop_size, dtype=np.int64) * row_len
    offsets = offsets.astype(np.int32).reshape(pop_size, *([1] * block_rank))
    return evoxc.constant(evoxc.from_numpy(np.ascontiguousarray(offsets)))


def _column_index(
    rows: evoxc.SymbolicTensor, attr_width: int
) -> evoxc.SymbolicTensor:
    """Widen a ``(P, N, K)`` connection-row index to every attribute column.
    Args:
        rows: ``(P, N, K) int32`` connection **row** index of each window slot.
        attr_width: ``CA``, the number of attribute columns of one connection row.
    Returns:
        ``(P, N, K, CA) int32`` index into the flattened ``(P, E * CA)`` attribute table,
        i.e. ``row * CA + column``.
    """
    scaled = evoxc.reshape(evoxc.multiply(rows, _i32(attr_width)), (*rows.shape, 1))
    column = evoxc.constant(
        evoxc.from_numpy(
            np.ascontiguousarray(
                np.arange(attr_width, dtype=np.int32).reshape(1, 1, 1, attr_width)
            )
        )
    )
    return evoxc.add(scaled, column)


def _column_mask(present: evoxc.SymbolicTensor) -> evoxc.SymbolicTensor:
    """Widen a ``(P, N, K)`` slot mask to the attribute columns of a connection row.

    The trailing ``1`` axis is left for the consumers to broadcast: ``logical_and`` against the
    ``(P, N, K, CA)`` index and ``select`` against the gathered ``(P, N, K, CA)`` window both
    broadcast it, so no explicit ``broadcast_to`` op is needed.

    Args:
        present: ``(P, N, K) bool`` liveness of each window slot.

    Returns:
        ``(P, N, K, 1) bool`` mask, broadcasting over the attribute columns.
    """
    return evoxc.reshape(present, (*present.shape[:3], 1))


def _gather_prepared(
    x: evoxc.SymbolicTensor,
    flat_index: evoxc.SymbolicTensor,
    present: evoxc.SymbolicTensor,
    block: tuple[int, ...],
    fill: float | None,
) -> evoxc.SymbolicTensor:
    """Gather one already-indexed window out of a batch-first ``(P, R)`` table.
    The per-level half of :meth:`BaseGenome._gather_slots`: three ops (flatten the table,
    gather, reshape back) once the index table of :func:`_slot_index` exists, plus one fill
    ``select`` when the caller asks for one.
    Args:
        x: ``(P, R)`` source table (node values, or a reshaped attribute payload).
        flat_index: ``(P * prod(block),)`` flat index from :func:`_slot_index`.
        present: ``(P, *block) bool`` liveness mask from :func:`_slot_index` (a broadcastable
            ``(P, *block[:-1], 1)`` mask is accepted too, so an attribute gather needs no
            explicit widening op).
        block: The window shape the result is reshaped to.
        fill: Value written where the slot is absent, or ``None`` to leave the gathered value
            in place — a value gather does not need the fill, because the slots it would
            overwrite are exactly the ones its liveness mask marks dead and the aggregation
            neutralises, whereas an attribute gather must fill NaN for the gene to clean.
    Returns:
        ``(P, *block)`` tensor with ``x``'s dtype.
    """
    pop_size, row_len = (int(size) for size in x.shape)
    values = evoxc.gather(evoxc.reshape(x, (pop_size * row_len,)), flat_index, axis=0)
    values = evoxc.reshape(values, (pop_size, *block))
    if fill is not None:
        # After the reshape, so the trailing-``1`` mask of ``_column_mask`` broadcasts without
        # an explicit widening op.
        values = evoxc.select(present, values, enp.full((), float(fill), dtype=x.dtype))
    return values
def _iinf(dtype: Any = evoxc.int32) -> evoxc.SymbolicTensor:
    """Rank-0 ``I_INF`` — the padding sentinel of every key array, left to broadcast.

    Deliberately *not* materialised at the operand's shape: ``enp.full(shape, I_INF)`` embeds a
    full-size array as a graph constant (``(P, N, K, CA)`` is 3.2 MB at ``P=1000``), which
    ``evoxc`` warns about ("Embedding a tensor ... as a graph constant") and which the numpy
    backend copies on every run. Every use here is a comparison (``not_equal``/``equal``), and
    those broadcast a rank-0 operand, so the constant stays 4 bytes and the op count is
    unchanged.

    Args:
        dtype: Integer dtype of the result.

    Returns:
        Rank-0 tensor holding ``I_INF``.
    """
    return enp.full((), I_INF, dtype=dtype)
