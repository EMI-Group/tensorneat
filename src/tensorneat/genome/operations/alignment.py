"""Exact-key homologue alignment, shared by crossover and compatibility distance.

Both operators ask the same question about two batches of parents — *which gene of parent 1
has a homologue in parent 2?* — so the answer is computed here once, as a **rank merge**:

1. each gene row is reduced to one sortable integer *identity* (:func:`row_identity`) —
   for a gene with one or two ``int32`` key columns the columns are packed side by side,
   which is injective, not merely a hash;
2. the two parents' identities are concatenated and ``argsort``-ed **once**, so inside a
   group of equal identities every parent-1 row sorts before every parent-2 row;
3. each gene's rank in that sorted order is read back as ``argsort(order)`` — the inverse
   permutation of the sort — so no scatter and no ``(P, 2M)`` match table is needed;
4. a parent-1 gene has a homologue exactly when **the element right after it in sorted
   order is a parent-2 row carrying the same identity**: one clamped rank increment plus
   two row gathers.

Homology is exact equality across the whole key-column tuple (``in_key``/``out_key`` plus
``historical_marker`` when the gene declares one), so the table is independent of the gene
family and of the row a gene happens to occupy — parent 2's rows may be in any order.
That is ``O(M log M)`` per individual in one batch-wide kernel over the population axis —
no Python loop over individuals, no ``vmap``, no ``scan``/``while_loop``/``cond``.

A gene with more than two key columns (``OriginConn``'s ``historical_marker``) cannot be
packed into one 64-bit identity, so :func:`row_identity` packs the **first two** columns —
the pair that identifies a gene, see the precondition below — and
:func:`find_homologous` **confirms** every candidate against the remaining columns, which
keeps homology equal to whole-tuple key-row equality.

The one precondition of the rank merge is the payload's own uniqueness invariant: *the first
two key columns are unique within one parent*.  It holds for both gene families the NEAT
payload ships — a node gene has a single key column (the node key, unique per genome) and a
connection gene's first two columns are its ``(in_key, out_key)`` pair, which
``execute_mutation`` refuses to duplicate and ``initialize`` builds once — and it is what
makes "the successor in sorted order" the *one* candidate partner of a gene.  A payload
that repeats an ``(in_key, out_key)`` pair would leave the pair's rows indistinguishable,
and homology there is undefined rather than merely expensive.

Two contract details drive the implementation, both about cost rather than maths:

* ``evoxc.gather`` follows numpy ``take`` semantics, so a per-individual row read is a flat
  1-D gather with ``row * row_len + column`` indices (that is what `gather_attr_rows`
  builds).  :func:`~tensorneat.common.tools.flat_gather` pays for escape checks and a fresh
  ``arange`` on every call; every index here is built in range by construction — a clamped
  rank, or a row already redirected by ``select`` — so the plain op is enough and the
  ``arange`` block is hoisted per row length (:func:`~tensorneat.common.tools.batch_offsets`).
* the reference interpreter costs ~80 µs per traced op regardless of element count, so op
  count *is* the budget: the merge is two sorts, two row gathers and the masks around them,
  and every ``select``/``not_equal`` here is one op over the whole ``(P, M)`` block rather
  than one op per attribute column.

Nothing here is differentiable (neither operator enters the ``grad`` graph); the sort and the
``int32`` index arithmetic are integer ops only.

:func:`key_rows` sits beside the merge because it is the same contract's front door: the
alignment is written against ``(P, M, K)`` key columns, so a caller with a two-dimensional
node-key table normalises it with that one helper instead of re-deriving the rank rule.
"""

import evoxc
import evoxc.numpy as enp

from ...common import I_INF, ConfigError, batch_offsets

#: Bits one key column occupies in the packed row identity.  A key is a non-negative
#: ``int32`` — the payload contract says keys live in ``[0, key_span)`` and ``I_INF =
#: 2**31 - 1`` is the pad sentinel — so 31 bits hold every value a column can carry.
_COLUMN_BITS = 31

#: Low bits of the packed identity reserved for the parent tag.
_TAG_BITS = 2

#: Tag bit of a parent-2 row.  Parent 1's tag is ``0``, so its rows need no extra op.
_TAG_PARENT_2 = 1

#: Number of key columns packed into the identity.  ``2 * 31`` bits plus the tag bits fill
#: an ``uint64`` exactly, so the identity is injective over the whole legal key domain; the
#: columns of a wider gene are confirmed by :func:`_confirm_rows` instead.
_PACKED_KEY_WIDTH = 2


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


def key_rows(keys: evoxc.SymbolicTensor) -> evoxc.SymbolicTensor:
    """Normalise a gene key table to the ``(P, M, K)`` key-column block.

    This is the key-table rank normaliser shared by crossover and compatibility distance:
    the alignment is written against three-dimensional key columns, while the payload keeps
    a node key in a single ``int32`` column, so ``node_keys`` is ``(P, N)`` while
    ``conn_keys`` is already ``(P, E, CK)``.  The implicit key axis is added here rather
    than special-cased inside the kernels.

    Args:
        keys (evoxc.SymbolicTensor): ``(P, M)`` or ``(P, M, K)`` ``int32`` key table,
            ``I_INF`` padded.

    Returns:
        evoxc.SymbolicTensor: ``(P, M, K)`` view of ``keys`` — the very same tensor when it
        already carries a key axis, otherwise a ``(P, M, 1)`` view of the single key column.

    Raises:
        ConfigError: If the table is neither 2-D nor 3-D (both ranks are static at trace
            time, so this is a host-side validation of a build-time property).  It is a
            :class:`~tensorneat.common.errors.TensorNEATError` subclass.
    """
    rank = len(keys.shape)
    if rank == 3:
        return keys
    if rank != 2:
        raise ConfigError(
            "gene key tables must be (P, M) or (P, M, K), "
            f"got shape {tuple(keys.shape)}"
        )
    pop_size, count = keys.shape
    return evoxc.reshape(keys, (pop_size, count, 1))


def row_identity(keys: evoxc.SymbolicTensor) -> evoxc.SymbolicTensor:
    """Reduce one key row to the sortable integer that stands for its key tuple.

    Args:
        keys (evoxc.SymbolicTensor): ``(P, M, K)`` ``int32`` key columns of one gene
            family, ``I_INF`` padded.

    Returns:
        evoxc.SymbolicTensor: ``(P, M)`` ``uint64`` row identities, low :data:`_TAG_BITS`
        bits clear (they hold the parent tag, see :func:`find_homologous`).

    Notes:
        Packing is ``column 0 << 31 | column 1``, shifted up by :data:`_TAG_BITS`: a key is
        a non-negative ``int32`` and ``I_INF = 2**31 - 1`` is the largest value one column
        can carry, so the pack is **injective over the whole legal key domain** — pad
        sentinel included — and two distinct key prefixes can never share an identity.  The
        fold is done in ``uint64`` because the top column's bits reach the sign bit of a
        signed ``int64``; unsigned shifts wrap cleanly there, so the identity is
        deterministic on every backend.

        Columns past :data:`_PACKED_KEY_WIDTH` do not fit one 64-bit integer, so they are
        left out of the sort key: they make the identity a *prefix* of the key tuple, and
        :func:`find_homologous` confirms every candidate against them with
        :func:`_confirm_rows`.  That is exact as long as the packed prefix is unique per
        parent, which the payload guarantees (a node gene has one key column; a connection
        gene's first two columns are its unique ``(in_key, out_key)`` pair).
    """
    pop_size, count = keys.shape[0], keys.shape[1]
    width = keys.shape[2]
    fold = evoxc.uint64
    if width == 1:
        identity = evoxc.cast(evoxc.reshape(keys, (pop_size, count)), fold)
    else:
        identity = evoxc.cast(_column(keys, 0), fold)
        shift = _scalar(_COLUMN_BITS, fold)
        for index in range(1, min(width, _PACKED_KEY_WIDTH)):
            identity = evoxc.bitwise_or(
                evoxc.bitwise_left_shift(identity, shift),
                evoxc.cast(_column(keys, index), fold),
            )
    return evoxc.bitwise_left_shift(identity, _scalar(_TAG_BITS, fold))


def _pad_identity(width: int) -> int:
    """The row identity of a padded row, i.e. of ``width`` ``I_INF`` key columns.

    Args:
        width (int): Static number of key columns of the gene family.

    Returns:
        int: The identity :func:`row_identity` produces for an absent row, computed on the
        host with the same arithmetic — so one ``not_equal`` separates parent 1's padding
        from a real gene instead of a per-column comparison.

    Notes:
        This is the host mirror of :func:`row_identity` on a plain-Python key tuple.  It has
        to stay in step with the traced pack (same stride, same reserved tag bits), or
        padding rows would stop being recognised; both read
        :data:`_COLUMN_BITS`/:data:`_TAG_BITS`, so only a change of *formula* can make them
        diverge.
    """
    identity = I_INF
    for _ in range(1, min(width, _PACKED_KEY_WIDTH)):
        identity = (identity << _COLUMN_BITS) | I_INF
    return identity << _TAG_BITS


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

    Notes:
        The per-individual stride is folded into the row term *before* the column term is
        broadcast, so exactly one of the three index ops works on the full ``(P, M, width)``
        block; the other two run on the ``(P, M, 1)`` row block.  Element volume, not op
        count, is what a wide index block costs on the reference interpreter.
    """
    offsets = batch_offsets(pop_size, count * width)
    base = evoxc.add(
        evoxc.multiply(enp.reshape(rows, (pop_size, count)), _scalar(width, evoxc.int32)),
        offsets,
    )
    return evoxc.add(
        enp.reshape(base, (pop_size, count, 1)),
        enp.arange(0, width, dtype=evoxc.int32),
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
        Only needed when the gene has more key columns than :data:`_PACKED_KEY_WIDTH` packed
        into :func:`row_identity`; with at most two key columns the identity is injective
        and this pass would always agree.  The packed columns are skipped — they match by
        construction — so a three-column gene pays one gather plus one ``equal`` per spare
        column, folded with ``logical_and`` rather than an innermost-axis reduce (a reduce
        over the column axis costs 10-20 ordinary ops on the reference interpreter).
    """
    pop_size, count, width = keys2.shape
    safe = evoxc.select(candidates, rows, _scalar(0, evoxc.int32))
    flat = evoxc.reshape(keys2, (pop_size * count * width,))
    echoed = evoxc.gather(flat, _flat_row_index(safe, width, pop_size, count), axis=0)
    identical = None
    for index in range(_PACKED_KEY_WIDTH, width):
        column_equal = evoxc.equal(_column(echoed, index), _column(keys1, index))
        identical = (
            column_equal
            if identical is None
            else evoxc.logical_and(identical, column_equal)
        )
    return identical


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
        One ``argsort`` and two row gathers carry the whole merge — the sorted identity
        sequence is *gathered* through the permutation instead of being sorted a second
        time, and the rank of each gene is the ``argsort`` of the permutation.  "Immediately
        after me" is the right successor to look at because the **parent tag lives in the
        low bits of the sort key** (:func:`row_identity` leaves them clear and parent 2's
        rows set them), so inside a group of equal identities parent 1's row is strictly
        smaller than parent 2's and the ordering does not depend on the sort being stable —
        which is what lets the sort run unstable, 4x faster on the reference interpreter.

        The successor's identity is therefore tested against *own identity with parent 2's
        tag set*: an equal answer means the successor is a parent-2 row holding exactly this
        key tuple, and it needs no separate "is it a parent-2 row" comparison.  Two padded
        rows also get different tags, so the tag alone keeps them apart; the explicit
        padding-identity test is what keeps a padded parent-1 row from reporting a row at
        all.

        The successor index is clamped to the last position.  Only the gene with the largest
        identity can ever be clamped, and then the successor is the gene itself, whose tag
        differs from the tag being tested for — so the clamp cannot invent a homologue.

        A gene family with more key columns than :data:`_PACKED_KEY_WIDTH` gets one extra
        pass (:func:`_confirm_rows`) because its identity covers only the packed prefix.
    """
    pop_size, count = keys1.shape[0], keys1.shape[1]
    width = keys1.shape[2]
    tag = _scalar(_TAG_PARENT_2, evoxc.uint64)
    identity1 = row_identity(keys1)
    identity2 = evoxc.bitwise_or(row_identity(keys2), tag)
    merged = evoxc.concatenate((identity1, identity2), axis=1)

    order = evoxc.cast(evoxc.argsort(merged, axis=-1), evoxc.int32)
    ranks = evoxc.cast(evoxc.argsort(order, axis=-1), evoxc.int32)
    successors = evoxc.minimum(
        evoxc.add(evoxc.slice(ranks, (0, 0), (pop_size, count)), _scalar(1, evoxc.int32)),
        _scalar(2 * count - 1, evoxc.int32),
    )
    offsets = batch_offsets(pop_size, 2 * count)
    # Position ``rank[i]`` of ``order`` is the merged row holding gene ``i``, so position
    # ``rank[i] + 1`` of ``order`` is the successor's row and the same position of
    # ``gather(merged, order)`` is the successor's identity.
    matched = evoxc.equal(
        _gather_rows(_gather_rows(merged, order, offsets), successors, offsets),
        evoxc.bitwise_or(identity1, tag),
    )
    rows = evoxc.subtract(
        _gather_rows(order, successors, offsets), _scalar(count, evoxc.int32)
    )
    if width > _PACKED_KEY_WIDTH:
        matched = evoxc.logical_and(matched, _confirm_rows(keys1, keys2, rows, matched))
    present = evoxc.logical_and(
        matched,
        evoxc.not_equal(identity1, _scalar(_pad_identity(width), evoxc.uint64)),
    )
    return evoxc.select(present, rows, _scalar(I_INF, evoxc.int32))


__all__ = ["find_homologous", "gather_attr_rows", "key_rows", "row_identity"]
