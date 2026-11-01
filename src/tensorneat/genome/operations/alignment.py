"""Key alignment shared by crossover and compatibility distance.

Both operators answer the same question about two batches of parents — *which gene of
parent 1 has a homologue in parent 2?* — and both must answer it without an ``O(M^2)``
scan, so the alignment lives here once and each operator consumes the homologue table.

The exact-match alignment is a **sorted merge**: one row hash per gene
(:func:`~tensorneat.genome.gene.base.hash_int32`, the legacy fold over the ``int32`` key
columns), one ``argsort`` over the concatenation of both parents' hashes with a tag that
orders a parent-1 gene before its parent-2 counterpart, and one adjacency test over the
sorted block.  A gene present in both parents therefore becomes an adjacent
``(parent 1, parent 2)`` pair — the batched equivalent of ``np.intersect1d`` — and the
hash collision window is closed by comparing the full key rows of the surviving pairs.
That is ``O(M log M)`` per individual and one batch-wide kernel over the population axis:
no Python loop over individuals, no ``vmap``, no ``scan``/``while_loop``/``cond``.

Nothing here is differentiable (the operators never enter the ``grad`` graph); the sort
and the ``int32`` index arithmetic are integer ops only.
"""

import evoxc
import evoxc.numpy as enp

from ...common import I_INF, flat_gather, masked_flat_gather, masked_flat_scatter
from ..gene.base import hash_int32

#: Tag values of the sorted-merge key: the low two bits of ``hash * 4 + tag``.
#: Invalid rows carry the low tags so they can never satisfy a ``2 -> 3`` adjacency,
#: which is what keeps ``I_INF`` padding out of the homologue table without an extra
#: validity gather; a parent-1 gene sorts before its parent-2 counterpart.
_TAG_INVALID_1 = 0
_TAG_INVALID_2 = 1
_TAG_PARENT_1 = 2
_TAG_PARENT_2 = 3

#: Bit width of the tag suffix in the composite sort key.
_TAG_BITS = 2


def _column(x: evoxc.SymbolicTensor, index: int) -> evoxc.SymbolicTensor:
    """Read one static column of the last axis of a traced tensor.

    Args:
        x: Tensor whose last axis is the column axis.
        index: Static (non-negative) column index.

    Returns:
        ``x.shape[:-1]``-shaped view of column ``index``.

    Notes:
        Written as a kept-axis slice plus ``reshape`` like
        :func:`~tensorneat.genome.gene.base.column`, because ``evoxc`` rejects
        ``Ellipsis``/``None`` static indexing and an axis-dropping integer index
        is not a legal traced read.
    """
    rank = len(x.shape)
    slicer = (slice(None),) * (rank - 1) + (slice(index, index + 1),)
    return evoxc.reshape(x[slicer], tuple(x.shape[:-1]))


def _scalar(value: int, dtype) -> evoxc.SymbolicTensor:
    """Rank-0 constant of ``dtype`` (a bare literal would widen an int32 tensor).

    Args:
        value: Static integer baked into the trace.
        dtype: ``evoxc`` dtype of the result.

    Returns:
        Rank-0 tensor inside the active trace.
    """
    return enp.full((), value, dtype=dtype)


def find_homologous(
    keys1: evoxc.SymbolicTensor, keys2: evoxc.SymbolicTensor
) -> evoxc.SymbolicTensor:
    """Locate, for every gene of parent 1, its homologous row in parent 2.

    Homology is exact equality across the whole key-column tuple (``in_key``/``out_key``
    plus ``historical_marker`` when the gene declares one), so the table is independent of
    the gene family and of the row a gene happens to occupy.

    Args:
        keys1: ``(P, M, K) int32`` key columns of parent 1, ``I_INF`` padded.
        keys2: ``(P, M, K) int32`` key columns of parent 2, ``I_INF`` padded. Both parents
            must share ``M`` and ``K`` — the payload shapes are static at build time.

    Returns:
        ``(P, M) int32`` row index into ``keys2`` per gene of parent 1, with ``I_INF``
        where parent 2 has no homologue and for parent 1's padded rows.

    Raises:
        evoxc.ShapeError: Propagated unchanged if the two parents' shapes differ.

    Notes:
        The merge key is ``hash * 4 + tag`` in ``int64`` (a ``uint32`` hash shifted left by
        two bits), so one ``argsort`` orders the ``2M`` candidate rows by hash with the
        parent-1 row first inside every hash group; a ``(2, 3)`` tag adjacency with equal
        hash is a candidate pair, and the candidate survives only if the full key rows are
        equal. Every gather is masked before it reaches the backend, because unmasked
        positions legitimately hold columns outside the gathered axis (``I_INF`` padding
        and parent-2 columns) which the numpy backend rejects instead of clamping.
    """
    pop_size, count = keys1.shape[0], keys1.shape[1]
    key_width = keys1.shape[2]
    row_width = count * key_width

    hash1 = hash_int32(keys1)
    hash2 = hash_int32(keys2)
    valid1 = evoxc.not_equal(_column(keys1, 0), _scalar(I_INF, evoxc.int32))
    valid2 = evoxc.not_equal(_column(keys2, 0), _scalar(I_INF, evoxc.int32))
    tag1 = evoxc.select(valid1, _scalar(_TAG_PARENT_1, evoxc.int64), _scalar(_TAG_INVALID_1, evoxc.int64))
    tag2 = evoxc.select(valid2, _scalar(_TAG_PARENT_2, evoxc.int64), _scalar(_TAG_INVALID_2, evoxc.int64))
    composite = evoxc.concatenate(
        (
            evoxc.add(evoxc.bitwise_left_shift(evoxc.cast(hash1, evoxc.int64), _scalar(_TAG_BITS, evoxc.int64)), tag1),
            evoxc.add(evoxc.bitwise_left_shift(evoxc.cast(hash2, evoxc.int64), _scalar(_TAG_BITS, evoxc.int64)), tag2),
        ),
        axis=1,
    )
    order = evoxc.cast(evoxc.argsort(composite, axis=-1, stable=True), evoxc.int32)
    sorted_composite = flat_gather(composite, order)

    pair_count = 2 * count - 1
    left = evoxc.slice(sorted_composite, (0, 0), (pop_size, pair_count))
    right = evoxc.slice(sorted_composite, (0, 1), (pop_size, pair_count))
    tag_mask = _scalar((1 << _TAG_BITS) - 1, evoxc.int64)
    left_tag = evoxc.bitwise_and(left, tag_mask)
    right_tag = evoxc.bitwise_and(right, tag_mask)
    same_hash = evoxc.equal(
        evoxc.bitwise_right_shift(left, _scalar(_TAG_BITS, evoxc.int64)),
        evoxc.bitwise_right_shift(right, _scalar(_TAG_BITS, evoxc.int64)),
    )
    candidate = evoxc.logical_and(
        evoxc.logical_and(
            evoxc.equal(left_tag, _scalar(_TAG_PARENT_1, evoxc.int64)),
            evoxc.equal(right_tag, _scalar(_TAG_PARENT_2, evoxc.int64)),
        ),
        same_hash,
    )

    # Column of parent 1 / parent 2 behind every sorted adjacency; only the positions
    # selected by ``candidate`` are ever gathered, the rest are masked out.
    parent1_col = evoxc.slice(order, (0, 0), (pop_size, pair_count))
    parent2_col = evoxc.subtract(
        evoxc.slice(order, (0, 1), (pop_size, pair_count)), _scalar(count, evoxc.int32)
    )
    columns = enp.reshape(enp.arange(0, key_width, dtype=evoxc.int32), (1, 1, key_width))
    flat1 = enp.reshape(keys1, (pop_size, row_width))
    flat2 = enp.reshape(keys2, (pop_size, row_width))
    gate = enp.reshape(candidate, (pop_size, pair_count, 1))
    rows1 = masked_flat_gather(
        flat1,
        evoxc.add(enp.reshape(evoxc.multiply(parent1_col, _scalar(key_width, evoxc.int32)), (pop_size, pair_count, 1)), columns),
        mask=gate,
        fill=0,
    )
    rows2 = masked_flat_gather(
        flat2,
        evoxc.add(enp.reshape(evoxc.multiply(parent2_col, _scalar(key_width, evoxc.int32)), (pop_size, pair_count, 1)), columns),
        mask=gate,
        fill=0,
    )
    identical = evoxc.reduce_min(evoxc.cast(evoxc.equal(rows1, rows2), evoxc.int32), axes=-1)
    matched = evoxc.logical_and(candidate, evoxc.greater(identical, _scalar(0, evoxc.int32)))

    table = enp.full((pop_size, count + 1), I_INF, dtype=evoxc.int32)
    scattered = masked_flat_scatter(table, parent1_col, parent2_col, matched)
    return evoxc.slice(scattered, (0, 0), (pop_size, count))


__all__ = ["find_homologous"]
