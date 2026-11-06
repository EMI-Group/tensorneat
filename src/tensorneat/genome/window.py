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
* A population-wide fill is one element plus ``broadcast_to`` (:func:`fill`), never
  ``enp.full((P, ...))``: an ``enp.full`` whose shape carries the population axis bakes
  ~4 bytes per individual per graph constant, which the build and every dispatch pay for.
  A fill that is already tiny (rank-0 scalar, ``(1, W)`` row, ``(N,)`` table) keeps its
  direct ``enp.full``, because the broadcast would be an extra traced op for no bytes.
* ``select`` and ``logical_and`` broadcast, so the attribute-column mask is widened with a
  trailing-``1`` axis (``_column_mask``) instead of a ``broadcast_to`` op.
* The one *reduction* of the pass lives here too, as :func:`any_slot`: a reduction over the
  innermost window axis is far more expensive than the equivalent static slice fold (see
  ``_WINDOW_FOLD_LIMIT``), and it is the same static-window reasoning the index helpers use.

No state, no genome class: every function is a pure tensor transformation of its arguments, and
every static (``row_len``, ``attr_width``, shapes) is a Python value fixed at trace time.
"""

from typing import Any, NamedTuple

import evoxc
import evoxc.numpy as enp
import numpy as np

from ..common import I_INF, TensorNEATError
from ..common.schedule_accelerator import resolve_mode


def fill(
    shape: tuple[int, ...],
    value: Any,
    dtype: Any = evoxc.float32,
) -> evoxc.SymbolicTensor:
    """A tensor of ``shape`` whose every entry is ``value``, from a one-element constant.

    ``enp.full(shape, value)`` embeds ``prod(shape)`` values as a graph constant, and an
    axis of that shape is usually the population: ``(P, max_nodes, NA)`` float32 is 3.2 MB
    at ``P=1000``, twice the largest payload array. ``evoxc`` warns about such payloads,
    the build pays for them, and the numpy backend copies them on every dispatch. A
    one-element constant plus one ``broadcast_to`` produces exactly the same values for
    four bytes, so every population-wide fill in this node is spelled this way.

    The broadcast is an ordinary traced op (about as expensive as any other on the
    reference interpreter), which is why a fill that is *already* tiny — a rank-0 scalar, a
    ``(1, W)`` static row, a ``(N,)`` column table — keeps its direct ``enp.full``.

    Args:
        shape: Result shape; the population axis may appear in any position.
        value: Python scalar, or an ``evoxc`` dtype-compatible number.
        dtype: Result dtype.

    Returns:
        ``shape`` tensor holding ``value``; a rank-0 ``shape`` is the bare constant.
    """
    shape = tuple(int(size) for size in shape)
    if not shape:
        return enp.full((), value, dtype=dtype)
    return enp.broadcast_to(enp.full((1,) * len(shape), value, dtype=dtype), shape)


class SeedColumns(NamedTuple):
    """Accelerator spelling of the value seed's splice: gather the input columns into place.

    ``columns`` is a ``(max_nodes,) int32`` table of *column ids* -- the input columns in
    order, then the last input column repeated for every ``NaN`` row -- so
    ``evoxc.gather(seed, columns, axis=1)`` places all the input values at once, one
    dispatch for the whole ``(rows, max_nodes)`` block instead of one per part. The ``NaN``
    rows it repeats are then dropped by ``keep``, a ``(1, max_nodes) bool`` mask that
    broadcasts over the rows, and ``nan`` is the rank-0 ``NaN`` fill the masked cells take
    (derived once per pass, so a query emits no constant of its own). The index addresses
    *columns*, not rows, so this table is ``max_nodes`` entries -- 200 bytes at the shipped
    ``max_nodes=50`` -- whatever the row count of the batch.
    """

    columns: evoxc.SymbolicTensor
    keep: evoxc.SymbolicTensor
    nan: evoxc.SymbolicTensor


class SeedPad(NamedTuple):
    """Static plan for the ``NaN`` rows behind the input rows of a forward value seed.

    A pass seeds the input rows of ``values`` with the network inputs and leaves every other
    row ``NaN`` -- an uncomputed node, which is what makes a node without a live source keep
    ``NaN``. Two spellings of that splice ship; they produce bit-identical values and differ
    only in what a *compiled* backend lowers them to:

    * ``block`` -- the host spelling's ``(rows, max_nodes - num_inputs)`` ``NaN`` operand of
      ``evoxc.concatenate``. One traced op and by far the cheaper form on the numpy reference
      interpreter, whose budget is op count (a ``concatenate`` body is a plain copy) and where
      a per-column gather plus a ``select`` would be two ops for the same bytes. A compiled
      backend instead lowers the splice to *two* dispatches: one copy of the dynamic input
      rows and one copy of the constant block, the first of them a serialised ``slow_memcpy``.
    * ``columns`` -- the accelerator spelling's :class:`SeedColumns`, which is one dispatch:
      IREE fuses the column gather into the ``select`` that masks the ``NaN`` rows, so a
      policy query pays a single parallel elementwise kernel instead of two copies.

    Exactly one of the two is set, and both are ``None`` when the network has no non-input row
    (``max_nodes == num_inputs``), where the seed *is* the input block and no splice happens.
    """

    block: evoxc.SymbolicTensor | None
    columns: SeedColumns | None


def seed_pad(mode: str, num_inputs: int, max_nodes: int, rows: int) -> SeedPad:
    """Plan the value seed's ``NaN`` rows for one forward pass.

    Args:
        mode: Kernel formulation -- ``"host"`` (the numpy reference interpreter, where op
            count is the budget) or ``"accelerator"`` (a compiled device, where dispatch
            count is).
        num_inputs: ``I``, the input rows the seed must carry.
        max_nodes: ``N``, the rows of the value seed.
        rows: Rows the seed will carry -- the schedule's ``P``, or the tiled ``P * samples``.
            Only the host spelling needs it, because its ``NaN`` block is materialised.

    Returns:
        SeedPad: The ``block`` operand of ``mode == "host"``, the ``columns`` plan of
        ``mode == "accelerator"``, or neither when ``N == I``.

    Raises:
        tensorneat.common.errors.ConfigError: If ``mode`` is not a documented formulation.
    """
    width = int(max_nodes) - int(num_inputs)
    if width <= 0:
        return SeedPad(block=None, columns=None)
    if resolve_mode(mode) == "accelerator":
        # The dead cells must address an *in range* column: gather clamps nothing, and the
        # mask discards whatever these rows read, so the last input column is repeated.
        columns = np.concatenate(
            (
                np.arange(num_inputs, dtype=np.int32),
                np.full(width, max(int(num_inputs) - 1, 0), dtype=np.int32),
            )
        )
        keep = np.concatenate(
            (np.ones(num_inputs, dtype=bool), np.zeros(width, dtype=bool))
        ).reshape(1, max_nodes)
        return SeedPad(
            block=None,
            columns=SeedColumns(
                columns=evoxc.constant(evoxc.from_numpy(np.ascontiguousarray(columns))),
                keep=evoxc.constant(evoxc.from_numpy(np.ascontiguousarray(keep))),
                nan=enp.full((), float("nan"), dtype=evoxc.float32),
            ),
        )
    return SeedPad(
        block=fill((rows, width), float("nan"), dtype=evoxc.float32), columns=None
    )


def seed_values(seed: evoxc.SymbolicTensor, pad: SeedPad) -> evoxc.SymbolicTensor:
    """Splice ``pad``'s ``NaN`` rows behind the input rows of ``seed``.

    Args:
        seed: ``(rows, num_inputs)`` input block of the value seed, after any
            ``input_transform``.
        pad: The :class:`SeedPad` :func:`seed_pad` planned for this pass.

    Returns:
        ``(rows, max_nodes)`` value seed: the input rows followed by ``NaN``.
    """
    if pad.columns is not None:
        placed = evoxc.gather(seed, pad.columns.columns, axis=1)
        return enp.where(pad.columns.keep, placed, pad.columns.nan)
    if pad.block is not None:
        return evoxc.concatenate([seed, pad.block], axis=1)
    return seed


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


def by_samples(
    x: evoxc.SymbolicTensor, samples: int, pop_size: int
) -> evoxc.SymbolicTensor:
    """Repeat a batch-first tensor ``samples`` times along a new sample axis, ``p * samples + j``.
    A problem tiles its dataset or rollout over the population axis, so tiled row
    ``p * samples + j`` is the copy of individual ``p``. Everything a forward pass derives from
    the schedule or the payload (the gathered edge attributes, the prepared node columns, the
    liveness mask, the flat gather index) is therefore identical across those rows, and a
    ``P``-row derivation re-expanded here costs one copy per operand instead of being
    re-derived at ``P * samples`` rows. The copies are the exact values the tiled kernel used
    to derive for itself, so the level loop stays bit-identical rather than merely equivalent.
    The one operand that cannot be derived this way is the flat gather index, which has to
    carry the tiled row offset: :func:`_slot_index` expands the *clamped* index and adds the
    ``P * samples``-row offset table afterwards.
    ``samples == 1`` (a direct caller such as ``Pipeline.show`` or a unit test) and a tensor
    without the population axis (a shared ordering table, a rank-0 constant) pass through
    unchanged, so neither pays for an op it does not need.
    Args:
        x: Batch-first tensor whose axis 0 is the population, or any tensor without it.
        samples: Static sample count ``S`` the batch was tiled with.
        pop_size: Static population ``P`` of ``x``.
    Returns:
        ``(P * samples, *x.shape[1:])`` in the flat order ``p * samples + j``, or ``x``.
    """
    if samples == 1:
        return x
    shape = getattr(x, "shape", None)
    if not shape or int(shape[0]) != pop_size:
        return x
    trailing = tuple(int(size) for size in shape[1:])
    widened = enp.reshape(x, (pop_size, 1) + trailing)
    repeated = enp.broadcast_to(widened, (pop_size, samples) + trailing)
    return enp.reshape(repeated, (pop_size * samples,) + trailing)


def _slot_index(
    idx: evoxc.SymbolicTensor,
    mask: evoxc.SymbolicTensor,
    row_len: int,
    samples: int = 1,
) -> tuple[evoxc.SymbolicTensor, evoxc.SymbolicTensor]:
    """Flatten one window index table to a block-shaped gather index plus a liveness mask.
    ``evoxc.gather`` follows ``np.take`` semantics and splices the *whole* index shape into the
    axis it indexes, so the source table is indexed flat (``p * row_len + row``) and the flat
    index is then shaped like the window block: the gather returns that block shape directly,
    which is what keeps the per-level step to one gather and no reshape-back.
    Both results depend only on the schedule (``idx`` and ``mask``) and on the static
    ``row_len``, so :meth:`BaseGenome._hit_conn_attrs` derives them once per forward pass and
    every unrolled level reuses them instead of re-deriving them.
    ``samples`` is the static tiling factor of a problem-tiled batch: the clamp runs at the
    schedule's own ``P`` rows and only its result is expanded (:func:`by_samples`), while the
    offset table is baked for the ``P * samples`` tiled rows -- the index is the one operand
    whose value genuinely depends on the tiled row, so it keeps one ``add`` at that width
    instead of a clamp and a reshape-back per row group.
    Two facts make this two traced ops (:func:`evoxc.minimum` and one ``add``) instead of the
    four a mask-and-redirect form needs:
    * the liveness mask *is* ``mask``: the schedule guarantees ``in_src`` is ``I_INF`` at
      exactly the positions where ``in_valid`` is ``False`` (``validate_schedule`` checks it),
      so the ``idx != I_INF`` term of ``(idx != I_INF) & mask`` is redundant — and for the
      attribute gather the index is a widened row id, where the ``& mask`` of the old form
      already decided every dead position;
    * the padding sentinel only has to be brought *into range*, not redirected to a chosen
      row: every consumer of the gathered value masks the dead slots out (the aggregation
      neutral-fills them, ``live`` excludes them), so clamping ``I_INF`` to the last row is
      as good as redirecting it to row ``0`` while costing one elementwise op instead of a
      ``select`` plus the two comparisons feeding it.
    Args:
        idx: ``(P, *block) int32`` row indices into the source table's second axis; ``I_INF``
            (or a masked-out position) means "no slot".
        mask: ``(P, *block) bool`` liveness of each slot; returned as the ``present`` mask.
        row_len: ``R``, the source table's second-axis length.
        samples: Static tiling factor ``S`` of the batch the index is used in.

    Returns:
        ``(flat_index, present)``: the ``(P, *block)`` gather index into the flattened
        ``(P * R,)`` table, with every absent slot clamped to row ``R - 1`` of its own
        individual (the backend never sees an out-of-range index), and the ``(P, *block) bool``
        liveness mask.
    """
    pop_size = int(idx.shape[0])
    block = tuple(int(size) for size in idx.shape[1:])
    rows = pop_size * samples
    in_range = evoxc.minimum(idx, _i32(row_len - 1))
    flat = evoxc.add(
        by_samples(in_range, samples, pop_size), _row_offsets(rows, row_len, len(block))
    )
    return evoxc.reshape(flat, (rows, *block)), by_samples(mask, samples, pop_size)


def _row_offsets(pop_size: int, row_len: int, block_rank: int) -> evoxc.SymbolicTensor:
    """``p * row_len`` baked in as one constant, shaped to broadcast over ``block_rank`` axes.

    The per-individual offset of the flat gather is fully static — ``row_len`` is a Python int
    — so it is embedded as a constant instead of being rebuilt in-graph from an ``arange``
    (three fewer traced ops per window index, and one constant instead of two).

    Args:
        pop_size: ``P``, the number of rows the flattened table holds — the population, or
            the tiled ``P * samples`` rows of a problem-tiled batch.
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
    rows: evoxc.SymbolicTensor, attr_width: int, row_len: int
) -> evoxc.SymbolicTensor:
    """Widen a ``(P, N, K)`` connection-row index to every attribute column.
    ``attr_width == 1`` (the shipped families) is the flat row id itself: the widening is a
    single trailing-axis ``reshape``, because multiplying by ``1`` and adding the all-zero
    column would be two elementwise passes over ``(P, N, K)`` plus a constant for a value the
    gather already addresses. The multi-column form clamps the row id into the table *before*
    scaling it: ``I_INF * CA`` wraps ``int32``, and while the position is masked dead anyway, an
    out-of-range index is undefined behaviour for the compiled backends, so the clamp keeps
    every gather index in ``[0, P * R * CA)`` without a ``select``.
    Args:
        rows: ``(P, N, K) int32`` connection **row** index of each window slot.
        attr_width: ``CA``, the number of attribute columns of one connection row.
        row_len: ``R``, the number of connection rows of one individual.
    Returns:
        ``(P, N, K, CA) int32`` index into the flattened ``(P, E * CA)`` attribute table,
        i.e. ``row * CA + column``.
    """
    widened = evoxc.reshape(rows, (*rows.shape, 1))
    if attr_width == 1:
        return widened
    in_range = evoxc.minimum(rows, _i32(row_len - 1))
    scaled = evoxc.reshape(
        evoxc.multiply(in_range, _i32(attr_width)), (*rows.shape, 1)
    )
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
) -> evoxc.SymbolicTensor:
    """Gather one already-indexed window out of a batch-first ``(P, R)`` table.
    The per-level half of :meth:`BaseGenome._gather_slots`: exactly two ops (flatten the
    table, gather) once the index table of :func:`_slot_index` exists — the index already
    carries the window block shape, so nothing is reshaped back.
    There is **no fill**: a statically dead slot is clamped to an in-range row of its own
    individual by :func:`_slot_index`/``_column_index`` and keeps that row's value, which no
    consumer reads — the caller's per-slot liveness mask (``_StepStatics.present`` ANDed with
    the ``x == x`` test) excludes it and the aggregation neutral-fills every dead slot before
    reducing. Writing ``NaN`` into the dead slots instead would cost one ``(P, N, K)``
    ``select`` per unrolled level for values nobody observes.
    Args:
        x: ``(P, R)`` source table (node values, or a reshaped attribute payload).
        flat_index: ``(P, *block)`` gather index into the flattened table, from
            :func:`_slot_index`.
    Returns:
        ``(P, *block)`` tensor with ``x``'s dtype.
    """
    return evoxc.gather(
        evoxc.reshape(x, (int(x.shape[0]) * int(x.shape[1]),)), flat_index, axis=0
    )
#: Window width up to which the sliced ``logical_or`` fold replaces the innermost-axis
#: ``reduce_max`` of the "has at least one live slot" test.
#:
#: A reduction over the *innermost* axis is the expensive shape on the numpy backend: it
#: costs about as much as a whole unrolled level's worth of elementwise ops, and it barely
#: depends on the window width (measured on ``(P, 50, K) bool`` as the median of 7
#: ``evoxc.run``, single-threaded: ``P=1000`` -> 1.23 ms at ``K=4`` and 1.16 ms at ``K=32``,
#: ``P=4000`` -> 4.4 ms at ``K=4`` and 5.7 ms at ``K=32``), because its per-run cost
#: dominates. The ``K``-slice ``logical_or`` fold of :func:`any_slot` instead scales with
#: the traced op count at ~0.016 ms per op: 0.13 ms at ``K=4``, 0.63 ms at ``K=16``
#: (``P=1000``). The measured crossover: at ``P=1000`` the fold still wins at ``K=24`` and
#: loses at ``K=32``; at ``P=100`` the whole run is one fixed overhead per op, so the
#: crossover sits between ``K=8`` (0.156 vs 0.158 ms) and ``K=12`` (0.196 vs 0.159 ms).
#: ``K <= 8`` is therefore the width where the fold wins at *every* measured population,
#: and wider windows fall back to the reduction.
_WINDOW_FOLD_LIMIT = 8


def any_slot(mask: evoxc.SymbolicTensor) -> evoxc.SymbolicTensor:
    """Fold the static window axis of a mask to "does this node have any live slot".

    The exact equivalent of ``evoxc.reduce_max(mask, axes=-1)`` on a ``(P, N, K)`` bool
    block, written as ``K`` static slices folded with ``logical_or`` because that
    *innermost*-axis reduction is one of the most expensive ops per unrolled level (see
    :data:`_WINDOW_FOLD_LIMIT`). Both spellings are a boolean "any", so the result is
    bit-identical, and the sliced folding emits ``K`` slices plus ``K - 1`` ``logical_or``
    plus one reshape (``2K`` ops) — a slightly larger *op count* that is several times
    faster in wall clock: the objective is time, and the op budget only matters when it
    buys nothing.

    ``evoxc.slice`` takes per-axis ``start``/``lengths``, so each slice must spell out
    ``(0, 0, k)``; a bare scalar start broadcasts to every axis and would try to slice the
    node axis too. The fold stays rank-3 until the end (one reshape, not ``K`` of them),
    because ``logical_or`` broadcasts a ``(P, N, 1)`` operand onto another ``(P, N, 1)``.

    Args:
        mask: ``(P, N, K) bool`` per-slot liveness mask, ``K`` the static window width.

    Returns:
        ``(P, N) bool``: True where at least one window slot is live.

    Raises:
        TensorNEATError: If the window is empty (``K == 0``) — there is no slot to fold and
            a silent ``False`` would turn every node into a NaN carrier.
    """
    pop_size, num_nodes, width = (int(size) for size in mask.shape)
    if width <= 0:
        raise TensorNEATError(
            f"any_slot cannot fold an empty window (K={width}); max_in_degree must be >= 1"
        )
    if width > _WINDOW_FOLD_LIMIT:
        return evoxc.reduce_max(mask, axes=-1)
    folded = evoxc.slice(mask, (0, 0, 0), (pop_size, num_nodes, 1))
    for slot in range(1, width):
        folded = evoxc.logical_or(
            folded, evoxc.slice(mask, (0, 0, slot), (pop_size, num_nodes, 1))
        )
    return evoxc.reshape(folded, (pop_size, num_nodes))


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
