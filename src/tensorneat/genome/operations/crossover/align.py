"""The crossover's own exact-key homologue merge, and the row gather that consumes it.

Crossover asks one question of two parent batches — *which gene of parent 1 has a
homologue in parent 2?* — and the answer is a **rank merge**:

1. each gene row is reduced to one sortable integer *identity* (:func:`row_identity`) —
   for a gene with one or two ``int32`` key columns the columns are packed side by side,
   which is injective, not merely a hash;
2. the two parents' identities are concatenated and ``argsort``-ed **once, stably**, so
   inside a group of equal identities every parent-1 row sorts before every parent-2 row;
3. each gene's rank in that sorted order is read back as ``argsort(order)`` — the inverse
   permutation of the sort — so no scatter and no ``(P, 2M)`` match table is needed;
4. a parent-1 gene has a homologue exactly when **the element right after it in sorted
   order is a parent-2 row carrying the same identity**: one clamped rank increment plus
   two row gathers.

That is ``O(M log M)`` per individual in one batch-wide kernel over the population axis —
no Python loop over individuals, no ``vmap``, no ``scan``/``while_loop``/``cond``.

A gene with more than two key columns (``OriginConn``'s ``historical_marker``) cannot be
packed into one 64-bit identity, so :func:`row_identity` folds the spare columns and
:func:`find_homologous` **confirms** every candidate against the full key rows, which keeps
homology equal to whole-tuple key-row equality at any key width.

Two contract details drive the implementation, both about cost rather than maths:

* ``evoxc.gather`` follows numpy ``take`` semantics, so a per-individual row read is a flat
  1-D gather with ``row * row_len + column`` indices (that is what ``gather_attr_rows``
  builds).  :func:`~tensorneat.common.tools.flat_gather` pays for escape checks and a fresh
  ``arange`` on every call; every index here is built in range by construction — a clamped
  rank, or a row already redirected by ``select`` — so the plain op is enough and the
  ``arange`` block is hoisted per row length (:func:`~tensorneat.common.tools.batch_offsets`).
* the reference interpreter costs ~80 µs per traced op regardless of element count, so op
  count *is* the budget: the merge is two sorts, two row gathers and the masks around them,
  and every ``select``/``not_equal`` here is one op over the whole ``(P, M)`` block rather
  than one op per attribute column.

Nothing here is differentiable (crossover never enters the ``grad`` graph); the sort and the
``int32`` index arithmetic are integer ops only.

:func:`~tensorneat.genome.operations.alignment.key_rows` is imported rather than
re-implemented: the rank normaliser is the shared front door of this contract, and the
same merge is what the compatibility distance needs, so the two operators must not drift
apart on what a "key row" is.
"""

from typing import TYPE_CHECKING

import evoxc
import evoxc.numpy as enp

from ....common import I_INF, batch_offsets
from ..alignment import key_rows

if TYPE_CHECKING:  # pragma: no cover - typing only
    pass

#: Bits one key column occupies in the packed row identity.
_COLUMN_BITS = 32

#: Widest key table whose identity is provably injective (``2 * 32`` bits fit the domain).
_EXACT_KEY_WIDTH = 2

#: Mask that keeps one ``int32`` column's bit pattern inside the packed identity.
_COLUMN_MASK = (1 << _COLUMN_BITS) - 1

#: Modulus of the folded identity, i.e. the wrap width of an unsigned 64-bit fold.
_FOLD_MODULUS = 1 << 64


def _column(x: evoxc.SymbolicTensor, index: int) -> evoxc.SymbolicTensor:
    """Read one static column of the last axis of a traced tensor.

    Args:
        x (evoxc.SymbolicTensor): Tensor whose last axis is the column axis.
        index (int): Static (non-negative) column index.

    Returns:
        evoxc.SymbolicTensor: ``x.shape[:-1]``-shaped view of column ``index``.

    Notes:
        Written as a kept-axis slice plus ``reshape`` like
        :func:`~tensorneat.genome.gene.base.column`, because ``evoxc`` rejects
        ``Ellipsis``/``None`` static indexing and an axis-dropping integer index is not a
        legal traced read.
    """
    rank = len(x.shape)
    slicer = (slice(None),) * (rank - 1) + (slice(index, index + 1),)
    return evoxc.reshape(x[slicer], tuple(x.shape[:-1]))


def _scalar(value: int, dtype) -> evoxc.SymbolicTensor:
    """Rank-0 constant of ``dtype`` (a bare literal would widen an ``int32`` tensor).

    Args:
        value (int): Static integer baked into the trace.
        dtype: ``evoxc`` dtype of the result.

    Returns:
        evoxc.SymbolicTensor: Rank-0 tensor inside the active trace.
    """
    return enp.full((), value, dtype=dtype)


def row_identity(keys: evoxc.SymbolicTensor) -> evoxc.SymbolicTensor:
    """Reduce one key row to the sortable integer that stands for its key tuple.

    Args:
        keys (evoxc.SymbolicTensor): ``(P, M, K)`` ``int32`` key columns of one gene
            family, ``I_INF`` padded.

    Returns:
        evoxc.SymbolicTensor: ``(P, M)`` row identities — the key itself for a one-column
        table (``int32``), the two packed columns for a two-column table (``uint64``), and
        a folded 64-bit mix beyond that (``uint64``, confirmed afterwards).

    Notes:
        Packing is ``column 0 << 32 | column 1`` with every column masked to its own 32
        bits first: an ``int32`` is 32 bits wide, so the pack is **injective over the whole
        ``int32`` range** — negative keys and the ``I_INF`` pad sentinel included — and two
        distinct key rows can never share an identity.  The fold is done in ``uint64``
        because the top column's bits reach the sign bit of a signed ``int64``; unsigned
        shifts wrap cleanly there, so the identity is deterministic on every backend.

        A third column cannot fit one 64-bit integer.  Folding it shifts the accumulated
        bits out of the register, so the result is only a *candidate* identity and
        :func:`find_homologous` confirms it against the full key rows.
    """
    pop_size, count = keys.shape[0], keys.shape[1]
    width = keys.shape[2]
    if width == 1:  # the key itself is the identity; no cast, no mask, nothing to fold
        return evoxc.reshape(keys, (pop_size, count))
    fold = evoxc.uint64
    mask = _scalar(_COLUMN_MASK, fold)
    identity = evoxc.bitwise_and(evoxc.cast(_column(keys, 0), fold), mask)
    shift = _scalar(_COLUMN_BITS, fold)
    for index in range(1, width):
        identity = evoxc.bitwise_or(
            evoxc.bitwise_left_shift(identity, shift),
            evoxc.bitwise_and(evoxc.cast(_column(keys, index), fold), mask),
        )
    return identity


def _pad_identity(width: int) -> int:
    """The row identity of a padded row, i.e. of ``width`` ``I_INF`` key columns.

    Args:
        width (int): Static number of key columns of the gene family.

    Returns:
        int: The identity :func:`row_identity` produces for an absent row, folded on the
        host with the same arithmetic — so one ``not_equal`` separates parent 1's padding
        from a real gene instead of a per-column comparison.

    Notes:
        Values are folded modulo ``2**64`` exactly like the traced ``uint64`` shift, and a
        result at or above ``2**63`` is returned as the wrapped signed value so it can be
        written into an ``int64`` constant by :func:`find_homologous` when the family has a
        single key column.
    """
    identity = I_INF
    for index in range(1, width):
        identity = ((identity << _COLUMN_BITS) | (I_INF & _COLUMN_MASK)) % _FOLD_MODULUS
    return identity


def _gather_rows(
    source: evoxc.SymbolicTensor,
    rows: evoxc.SymbolicTensor,
    offsets: evoxc.SymbolicTensor,
) -> evoxc.SymbolicTensor:
    """Gather one row of a ``(P, R)`` source per individual: ``out[p, i] = source[p, rows[p, i]]``.

    Args:
        source (evoxc.SymbolicTensor): ``(P, R)`` batch-first source.
        rows (evoxc.SymbolicTensor): ``(P, M)`` ``int32`` row indices, every one already in
            ``[0, R)`` — this helper does not clamp, so a caller that can produce an
            escaping index must redirect it first.
        offsets (evoxc.SymbolicTensor): ``(P, 1)`` result of
            :func:`~tensorneat.common.tools.batch_offsets` for ``R``, hoisted by the caller
            so several gathers over the same source share one ``arange``.

    Returns:
        evoxc.SymbolicTensor: ``(P, M)`` gathered rows.
    """
    pop_size, row_len = source.shape
    flat = evoxc.reshape(source, (pop_size * row_len,))
    return evoxc.gather(flat, evoxc.add(rows, offsets), axis=0)


def _flat_row_index(
    rows: evoxc.SymbolicTensor, width: int, pop_size: int, count: int
) -> evoxc.SymbolicTensor:
    """Flat index block reading one ``width``-wide row per gene of a ``(P, M * width)`` payload.

    Args:
        rows (evoxc.SymbolicTensor): ``(P, M)`` ``int32`` row indices, every one already in
            ``[0, M)`` — the read is unmasked, so a caller with absent positions must have
            redirected them first.
        width (int): Static number of columns per row.
        pop_size (int): Static ``P``.
        count (int): Static ``M``.

    Returns:
        evoxc.SymbolicTensor: ``(P, M, width)`` ``int32`` flat indices
        ``row * width + column + p * (M * width)``.
    """
    offsets = enp.reshape(batch_offsets(pop_size, count * width), (pop_size, 1, 1))
    return evoxc.add(
        evoxc.add(
            evoxc.multiply(
                enp.reshape(rows, (pop_size, count, 1)), _scalar(width, evoxc.int32)
            ),
            enp.arange(0, width, dtype=evoxc.int32),
        ),
        offsets,
    )


def gather_attr_rows(
    attrs: evoxc.SymbolicTensor, rows: evoxc.SymbolicTensor
) -> evoxc.SymbolicTensor:
    """Pick one attribute row per individual out of parent 2's payload.

    Args:
        attrs (evoxc.SymbolicTensor): ``(P, M, A)`` attribute block of parent 2, NaN padded.
        rows (evoxc.SymbolicTensor): ``(P, M)`` ``int32`` rows into ``attrs``, every one
            already in ``[0, M)`` — an absent homologue must have been redirected by the
            caller's ``select`` (row ``0`` is the usual target) because the read itself is
            unmasked.

    Returns:
        evoxc.SymbolicTensor: ``(P, M, A)`` rows of ``attrs``, one row per gene of parent 1.

    Notes:
        A single flat gather per attribute **row** — not one gather per attribute column —
        so the cost is one op for the whole block.  The index is
        ``row * A + column + p * (M * A)``, the per-individual stride coming from
        :func:`~tensorneat.common.tools.batch_offsets`.
    """
    pop_size, count, width = attrs.shape
    flat = evoxc.reshape(attrs, (pop_size * count * width,))
    return evoxc.gather(flat, _flat_row_index(rows, width, pop_size, count), axis=0)


def _confirm_rows(
    keys1: evoxc.SymbolicTensor,
    keys2: evoxc.SymbolicTensor,
    rows: evoxc.SymbolicTensor,
    candidates: evoxc.SymbolicTensor,
) -> evoxc.SymbolicTensor:
    """Check proposed homologues against the full key rows, column for column.

    Args:
        keys1 (evoxc.SymbolicTensor): ``(P, M, K)`` key columns of parent 1.
        keys2 (evoxc.SymbolicTensor): ``(P, M, K)`` key columns of parent 2.
        rows (evoxc.SymbolicTensor): ``(P, M)`` proposed row in parent 2 per gene of parent
            1; entries that are not candidates are redirected to row ``0`` here.
        candidates (evoxc.SymbolicTensor): ``(P, M)`` bool, the rows worth confirming.

    Returns:
        evoxc.SymbolicTensor: ``(P, M)`` bool, True where parent 2's row holds exactly
        parent 1's key tuple.

    Notes:
        Only needed when :func:`row_identity` folded a third column away; with at most two
        key columns the identity is injective and this pass would always agree.
    """
    pop_size, count, width = keys2.shape
    safe = evoxc.select(candidates, rows, _scalar(0, evoxc.int32))
    flat = evoxc.reshape(keys2, (pop_size * count * width,))
    echoed = evoxc.gather(flat, _flat_row_index(safe, width, pop_size, count), axis=0)
    identical = evoxc.reduce_min(
        evoxc.cast(evoxc.equal(echoed, keys1), evoxc.int32), axes=-1
    )
    return evoxc.greater(identical, _scalar(0, evoxc.int32))


def find_homologous(
    keys1: evoxc.SymbolicTensor, keys2: evoxc.SymbolicTensor
) -> evoxc.SymbolicTensor:
    """Locate, for every gene of parent 1, its homologous row in parent 2.

    Exact match on the whole key-column tuple (``in_key``/``out_key`` plus
    ``historical_marker`` when the gene declares one), so the answer is independent of the
    row a gene happens to occupy — parent 2's rows may be in any order.

    Args:
        keys1 (evoxc.SymbolicTensor): ``(P, M, K)`` ``int32`` key columns of parent 1,
            ``I_INF`` padded.
        keys2 (evoxc.SymbolicTensor): ``(P, M, K)`` ``int32`` key columns of parent 2,
            ``I_INF`` padded. Both parents share ``M`` and ``K``: the payload shapes are
            static at build time (the caller validates that, see
            :meth:`~tensorneat.genome.operations.crossover.base.BaseCrossover.__call__`).

    Returns:
        evoxc.SymbolicTensor: ``(P, M)`` ``int32`` row index into ``keys2`` per gene of
        parent 1, with ``I_INF`` where parent 2 has no homologue and for parent 1's padded
        rows.

    Raises:
        evoxc.ShapeError: Propagated unchanged if the two parents' shapes differ.

    Notes:
        Two sorts and two row gathers carry the whole merge.  The stable sort is what makes
        "immediately after me" the right successor to look at: inside a group of equal
        identities parent 1's row comes first, so its successor is parent 2's row when a
        homologue exists and something with a *different* identity otherwise.

        The successor index is clamped to the last position.  Only the gene with the
        largest identity can ever be clamped, and its clamped successor identity differs
        from its own, so the clamp cannot invent a homologue; parent 1's padded rows are
        excluded explicitly by comparing the identity against the padding identity, which
        is why two padded rows never align with each other.

        A gene family with more than two key columns gets one extra pass
        (:func:`_confirm_rows`) because its identity is only a candidate match there.
    """
    pop_size, count = keys1.shape[0], keys1.shape[1]
    width = keys1.shape[2]
    identity1 = row_identity(keys1)
    identity2 = row_identity(keys2)
    merged = evoxc.concatenate((identity1, identity2), axis=1)

    order = evoxc.cast(evoxc.argsort(merged, axis=-1, stable=True), evoxc.int32)
    ranks = evoxc.cast(evoxc.argsort(order, axis=-1, stable=True), evoxc.int32)
    rank = evoxc.slice(ranks, (0, 0), (pop_size, count))
    successors = evoxc.minimum(
        evoxc.add(rank, _scalar(1, evoxc.int32)),
        _scalar(2 * count - 1, evoxc.int32),
    )
    offsets = batch_offsets(pop_size, 2 * count)
    # ``order`` is a permutation, so ``sorted`` is the identity sequence in sorted order:
    # position ``rank[i] + 1`` of it is the *successor's identity*, while position
    # ``rank[i] + 1`` of ``order`` is the successor's row in the merged batch.
    sorted_identity = _gather_rows(merged, order, offsets)
    successor_row = _gather_rows(order, successors, offsets)
    matched = evoxc.logical_and(
        evoxc.equal(_gather_rows(sorted_identity, successors, offsets), identity1),
        evoxc.greater_equal(successor_row, _scalar(count, evoxc.int32)),
    )
    rows = evoxc.subtract(successor_row, _scalar(count, evoxc.int32))
    if width > _EXACT_KEY_WIDTH:
        matched = evoxc.logical_and(matched, _confirm_rows(keys1, keys2, rows, matched))
    present = evoxc.logical_and(
        matched,
        evoxc.not_equal(identity1, _scalar(_pad_identity(width), identity1.dtype)),
    )
    return evoxc.select(present, rows, _scalar(I_INF, evoxc.int32))


__all__ = ["find_homologous", "gather_attr_rows", "key_rows", "row_identity"]
