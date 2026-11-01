"""Flat-gather plumbing of the window step: slot indices, column widening, prepared gather.

This module is the *index arithmetic* half of the forward pass — everything
:meth:`tensorneat.genome.base.BaseGenome._hit_conn_attrs` derives once per pass so an unrolled
level pays only for the values themselves. It sits beside ``base.py`` (which keeps the genome
class and the window step) so that file stays one job and under its line budget.

Op count, not element count, is the budget on the numpy backend (one traced op costs about as
much as eighty microseconds at ``P=1000``, whatever its shape), so every helper here is written
to emit the fewest ops that stay bit-identical:

* ``evoxc.gather`` has ``np.take`` semantics and splices the *whole* index shape into the axis
  it indexes, so the table is indexed flat (``p * row_len + row``) and the flat index is then
  *shaped* like the window block: the gather returns the block shape directly, which is what
  lets an unrolled level skip its reshape-back. The per-individual offset ``p * row_len`` is
  static, hence a single constant (``_row_offsets``) instead of an in-graph
  ``arange``/``multiply``/``reshape`` chain.
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
    """Flatten one window index table to a block-shaped gather index plus a liveness mask.
    ``evoxc.gather`` follows ``np.take`` semantics and splices the *whole* index shape into the
    axis it indexes, so the source table is indexed flat (``p * row_len + row``) and the flat
    index is then shaped like the window block: the gather returns that block shape directly,
    which is what keeps the per-level step to one gather and no reshape-back.
    Both results depend only on the schedule (``idx`` and ``mask``) and on the static
    ``row_len``, so :meth:`BaseGenome._hit_conn_attrs` derives them once per forward pass and
    every unrolled level reuses them instead of re-deriving them.
    Args:
        idx: ``(P, *block) int32`` row indices into the source table's second axis; ``I_INF``
            (or a masked-out position) means "no slot".
        mask: ``(P, *block) bool`` liveness of each slot.
        row_len: ``R``, the source table's second-axis length.
    Returns:
        ``(flat_index, present)``: the ``(P, *block)`` gather index into the flattened
        ``(P * R,)`` table, with every absent slot redirected to row ``0`` of its own
        individual (the backend never sees an out-of-range index), and the ``(P, *block) bool``
        liveness mask — block-shaped, which is the shape the per-level value gather uses to
        fill its dead slots with ``NaN``.
    """
    pop_size = int(idx.shape[0])
    block = tuple(int(size) for size in idx.shape[1:])
    present = evoxc.logical_and(enp.not_equal(idx, _iinf(idx.dtype)), mask)
    safe = evoxc.select(present, idx, enp.zeros((), dtype=idx.dtype))
    flat = evoxc.add(safe, _row_offsets(pop_size, row_len, len(block)))
    return evoxc.reshape(flat, (pop_size, *block)), present


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
    fill: float | evoxc.SymbolicTensor | None,
) -> evoxc.SymbolicTensor:
    """Gather one already-indexed window out of a batch-first ``(P, R)`` table.
    The per-level half of :meth:`BaseGenome._gather_slots`: two ops (flatten the table,
    gather) once the index table of :func:`_slot_index` exists — the index already carries
    the window block shape, so nothing is reshaped back — plus one fill ``select`` when the
    caller asks for one.
    Args:
        x: ``(P, R)`` source table (node values, or a reshaped attribute payload).
        flat_index: ``(P, *block)`` gather index into the flattened table, from
            :func:`_slot_index`.
        present: ``(P, *block) bool`` liveness mask from :func:`_slot_index` (a broadcastable
            ``(P, *block[:-1], 1)`` mask is accepted too, so an attribute gather needs no
            explicit widening op).
        fill: Value written where the slot is absent, or ``None`` to leave the gathered value
            in place. Both production callers pass ``NaN``: an *attribute* gather needs it so
            the gene's ``prepare`` sees a clean "absent row" sentinel, and a *value* gather
            needs it so the caller's liveness predicate is one ``x == x`` comparison — the
            slots the fill overwrites are exactly the ones the schedule marked dead, and a
            ``NaN`` there reduces to "not live" without a second mask. A rank-0 tensor is used
            verbatim instead of a Python float (which would materialise its own ``constant``
            op on every level); the value gather passes the per-pass NaN of
            :class:`_StepStatics` for exactly that reason.
    Returns:
        ``(P, *block)`` tensor with ``x``'s dtype.
    """
    values = evoxc.gather(
        evoxc.reshape(x, (int(x.shape[0]) * int(x.shape[1]),)), flat_index, axis=0
    )
    if fill is not None:
        # After the gather, so the trailing-``1`` mask of ``_column_mask`` broadcasts without
        # an explicit widening op.
        scalar = (
            fill
            if isinstance(fill, evoxc.SymbolicTensor)
            else enp.full((), float(fill), dtype=x.dtype)
        )
        values = evoxc.select(present, values, scalar)
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
