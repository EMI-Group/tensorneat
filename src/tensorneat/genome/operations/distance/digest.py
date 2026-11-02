"""Two-phase compatibility distance: one population digest, one merge per centre.

NEAT speciation measures a single centre genome against the whole population twice per
generation (centre reservation and member assignment), so the *same* population payload is
re-sorted ``2 * species_size`` times per generation.
This module splits that work: :func:`prepare_family` sorts one gene family of the population
**once** per generation (packed key composites, the attribute rows permuted into the same
order, and the valid-gene count), and :func:`match_rows` merges one centre genome against
that digest with a single sort of the ``2 * M`` concatenation — no per-pair ``O(M^2)``
matrix, no re-sort of the population and no key-row re-verification.

The injective composite is what makes the merge exact: a gene row is folded into one
``int64`` value ``pack``, positionally in base ``key_span`` for the node-key columns, with
the ``historical_marker`` column contributing one full 32-bit digit because it is an
arbitrary ``int32`` marker and ``key_span`` does not bound it.
The fold is injective, so *composite equality is key-row equality* and the legacy
hash-then-verify pair (two key-row gathers plus a ``reduce_min``) disappears.

The low two bits of the sort key carry a side tag — population ``3``, centre ``2`` — and each
side's padding rows collapse onto their own sentinel (``2**61`` for the centre,
``2**61 + 5`` for the population), so the tags are never adjacent by accident.
A homologous pair is then exactly an adjacent ``(centre, population)`` pair whose composites
differ by one, which is the merge's only comparison: ``right == left + 1``.
Both sides are *sorted ascending with the padding last*, so the population digest needs no
validity handling at all beyond the tag.

Determinism: the sort is not required to be stable (ties are padding rows, which can never
pair), so the cheap comparison sort is used; the emitted distances do not depend on the sort
order because the per-gene terms are scattered back to the centre's own row order before the
reduction (see :meth:`tensorneat.genome.operations.distance.base.BaseDistance.gene_measure`).
"""

from typing import NamedTuple

import evoxc
import evoxc.numpy as enp
import numpy as np

from ....common import I_INF, ConfigError, flat_gather, flat_scatter
from ..alignment import key_rows

#: Width of the side tag carried in the low bits of the composite sort key.
_TAG_BITS = 2

#: Side tag of a centre gene / of a population gene.  A homologous pair differs by exactly
#: ``_TAG_POPULATION - _TAG_CENTRE`` in the composite, which is the merge's whole test.
_TAG_CENTRE = 2
_TAG_POPULATION = 3

#: Padding sentinels of the two sides.  Distinct and far apart, so neither a
#: ``(padding, padding)`` nor a ``(padding, gene)`` adjacency can ever differ by one.
_PAD_CENTRE = 1 << 61
_PAD_POPULATION = (1 << 61) + 5

#: Weight of one ``historical_marker`` digit: the marker is an arbitrary ``int32``, and
#: ``2**32`` (its exclusive range width) keeps the fold injective without a sign bias.
_MARKER_BASE = 1 << 32

#: Exclusive bound of the packed composite.  The tag needs ``_TAG_BITS`` low bits and the
#: arithmetic happens in ``int64``, so the fold must stay below ``2**62``.
_COMPOSITE_LIMIT = 1 << (_TAG_BITS + 60)


class FamilyDigest(NamedTuple):
    """Prepared form of one gene family of a population (batch-first, static shapes).

    Attributes:
        keys (evoxc.SymbolicTensor): ``(P, M) int64`` packed key composites, ascending with
            the padding sentinel last.
        attrs (evoxc.SymbolicTensor): ``(P, M, A) float32`` attribute rows permuted into the
            order of :attr:`keys`; absent rows keep their ``NaN`` attributes.
        count (evoxc.SymbolicTensor): ``(P,) int32`` number of real genes per individual, i.e.
            the length of the valid prefix of :attr:`keys`.
    """

    keys: evoxc.SymbolicTensor
    attrs: evoxc.SymbolicTensor
    count: evoxc.SymbolicTensor


def _i32(value: int) -> evoxc.SymbolicTensor:
    """Build a rank-0 ``int32`` constant inside the active trace.

    Args:
        value: Static integer baked into the trace.

    Returns:
        Rank-0 ``int32`` tensor.

    Notes:
        Spelled out because an ``int32`` tensor plus a Python literal promotes to ``int64``
        in ``evoxc``, which would widen every index computation below.
    """
    return enp.full((), value, dtype=evoxc.int32)


def _i64(value: int) -> evoxc.SymbolicTensor:
    """Build a rank-0 ``int64`` constant inside the active trace.

    Args:
        value: Static integer baked into the trace.

    Returns:
        Rank-0 ``int64`` tensor — the dtype of every composite in this module.
    """
    return enp.full((), value, dtype=evoxc.int64)


def _column(table: evoxc.SymbolicTensor, index: int) -> evoxc.SymbolicTensor:
    """Read one static key column of a ``(P, M, K)`` table.

    Args:
        table: ``(P, M, K)`` key-column block.
        index: Static (non-negative) column index.

    Returns:
        ``(P, M)`` view of column ``index``.

    Notes:
        A kept-axis slice plus ``reshape`` because ``evoxc`` rejects an axis-dropping
        integer index and ``Ellipsis``/``None`` static indexing.
    """
    pop_size, count = int(table.shape[0]), int(table.shape[1])
    return evoxc.reshape(
        table[(slice(None), slice(None), slice(index, index + 1))], (pop_size, count)
    )


def _key_table(
    name: str, gene, keys: evoxc.SymbolicTensor
) -> evoxc.SymbolicTensor:
    """Normalise one payload key array to the ``(P, M, K)`` key-column block.

    Args:
        name: Operator name used in the error message.
        gene: Node or connection gene declaring ``key_width``.
        keys: ``(P, M)`` (one-column node key) or ``(P, M, K)`` key array.

    Returns:
        ``(P, M, K)`` key-column block.

    Raises:
        ConfigError: If the block's key width is not the gene's ``key_width`` — the digest
            is built and consumed with one gene family at a time, so a mismatched width is a
            configuration error rather than a silent reinterpretation.
    """
    table = key_rows(keys)
    width = int(table.shape[-1])
    if width != int(gene.key_width):
        raise ConfigError(
            f"{name}: {type(gene).__name__} declares {gene.key_width} key column(s) but the "
            f"payload key array carries {width}"
        )
    return table


def _weights(gene, key_span: int) -> tuple[int, ...]:
    """Positional fold weights of one gene family's key columns.

    Args:
        gene: Node or connection gene declaring ``fixed_attrs``.
        key_span: Static key bound ``S`` from the genome (host value).

    Returns:
        One ``int`` weight per key column, most significant first.

    Raises:
        ConfigError: If the resulting composite cannot be represented in ``int64`` below
            ``2**62`` — i.e. ``key_span`` is too large for the gene's key width.  The caller
            must then build the genome with a smaller ``key_span`` (a configuration error,
            never a silent approximation of the key equality).

    Notes:
        Node-key columns are bounded by ``key_span`` (the same invariant the execution
        schedule's dense key table relies on); the ``historical_marker`` column is an
        arbitrary ``int32`` and gets a full 32-bit digit, which keeps the fold injective
        without a sign bias because ``2**31 < 2**32``.
    """
    width = int(gene.key_width)
    marker = gene.key_index("historical_marker") if gene.uses_historical_marker else None
    weights = [1] * width
    tail = 1
    for index in range(width - 1, -1, -1):
        weights[index] = tail
        tail *= _MARKER_BASE if index == marker else int(key_span)
        if tail > _COMPOSITE_LIMIT:
            raise ConfigError(
                f"compatibility distance cannot pack {type(gene).__name__} key rows of "
                f"width {width} into an int64 composite at key_span={key_span}; construct "
                "the genome with a smaller key_span"
            )
    return tuple(weights)


def _pack(
    gene, table: evoxc.SymbolicTensor, key_span: int
) -> tuple[evoxc.SymbolicTensor, evoxc.SymbolicTensor]:
    """Fold one gene family's key rows into injective ``int64`` composites.

    Args:
        gene: Node or connection gene declaring the key width.
        table: ``(P, M, K) int32`` key-column block, ``I_INF`` padded.
        key_span: Static key bound ``S`` from the genome.

    Returns:
        ``(pack (P, M) int64, valid (P, M) bool)`` — the folded key value and the row's
        "real gene" mask (the leading key column is not ``I_INF``).

    Notes:
        Wide key tables are folded positionally: one weighted ``multiply`` against a rank-3
        constant followed by a ``reduce_sum`` over the key axis, which is exact because the
        reduction is integer.
        Absent rows are first zeroed as a whole, so the fold can neither overflow on the
        ``I_INF`` digits nor depend on whatever an absent row happens to hold — their
        composites are then replaced by the side's sentinel by the caller.
    """
    leading = _column(table, 0)
    valid = evoxc.not_equal(leading, _i32(I_INF))
    width = int(table.shape[2])
    if width == 1:
        # A one-column (node) key table folds to itself; its digits stay inside
        # ``[0, key_span)`` or the ``I_INF`` sentinel, so no scaling is needed.
        return evoxc.cast(leading, evoxc.int64), valid
    # ``_weights`` is called for its bound check, and the fold below is the same
    # most-significant-first positional sum spelled as ``2 * width`` ops: a ``(P, M, K)``
    # innermost-axis integer reduction costs an order of magnitude more than the
    # ``multiply``/``add`` pair per column it replaces.
    _weights(gene, key_span)
    marker = gene.key_index("historical_marker") if gene.uses_historical_marker else None
    zero = _i64(0)
    composite = None
    for index in range(width):
        digit = evoxc.select(valid, evoxc.cast(_column(table, index), evoxc.int64), zero)
        if composite is None:
            composite = digit
            continue
        base = _MARKER_BASE if index == marker else int(key_span)
        composite = evoxc.add(evoxc.multiply(composite, _i64(base)), digit)
    return composite, valid


def _gather_rows(
    attrs: evoxc.SymbolicTensor, order: evoxc.SymbolicTensor, attr_width: int
) -> evoxc.SymbolicTensor:
    """Permute attribute rows into a gene order.

    Args:
        attrs: ``(P, M, A)`` attribute rows.
        order: ``(P, M) int32`` permutation (row ``j`` of the result is row ``order[j]``).
        attr_width: Static attribute width ``A``.

    Returns:
        ``(P, M, A)`` permuted attribute rows.
    """
    pop_size, count = int(order.shape[0]), int(order.shape[1])
    flat = evoxc.reshape(attrs, (pop_size, count * attr_width))
    offsets = enp.reshape(
        evoxc.multiply(order, _i32(attr_width)), (pop_size, count, 1)
    )
    columns = enp.reshape(enp.arange(0, attr_width, dtype=evoxc.int32), (1, 1, attr_width))
    return flat_gather(flat, evoxc.add(offsets, columns))


def prepare_family(
    state, gene, keys: evoxc.SymbolicTensor, attrs: evoxc.SymbolicTensor, key_span: int
) -> FamilyDigest:
    """Sort one gene family of a population into a reusable digest.

    Args:
        state: Host configuration container (unused; kept for the operator call shape).
        gene: Node or connection gene declaring the key and attribute layout.
        keys: ``(P, M)`` or ``(P, M, K) int32`` key array of the population, ``I_INF``
            padded.
        attrs: ``(P, M, A) float32`` attribute array of the population, NaN padded.
        key_span: Static key bound ``S`` from the genome.

    Returns:
        FamilyDigest: the packed composites (population tag) in ascending order with the
        padding sentinel last, the attribute rows permuted into that order, and the number of
        real genes per individual.

    Notes:
        This is the phase that used to run once per distance call; it now runs once per
        generation, so its cost is amortised over ``2 * species_size`` measurements.
        The sort is *not* stable on purpose: ties are padding rows only (real keys are
        unique inside a genome), and the distances do not depend on the tie order.
    """
    table = _key_table("prepare", gene, keys)
    pack, valid = _pack(gene, table, key_span)
    tagged = evoxc.select(
        valid,
        evoxc.add(
            evoxc.multiply(pack, _i64(1 << _TAG_BITS)), _i64(_TAG_POPULATION)
        ),
        _i64(_PAD_POPULATION),
    )
    order = evoxc.cast(evoxc.argsort(tagged, axis=-1, stable=False), evoxc.int32)
    keys_sorted = flat_gather(tagged, order)
    attrs_sorted = _gather_rows(attrs, order, int(attrs.shape[-1]))
    count = evoxc.cast(
        evoxc.reduce_sum(evoxc.cast(valid, evoxc.int32), axes=1), evoxc.int32
    )
    return FamilyDigest(keys_sorted, attrs_sorted, count)


def match_rows(
    gene, keys: evoxc.SymbolicTensor, digest_keys: evoxc.SymbolicTensor, key_span: int
) -> evoxc.SymbolicTensor:
    """Locate, for every gene of a centre genome, its homologue row in the digest.

    Args:
        gene: Node or connection gene declaring the key width.
        keys: ``(P, M)`` or ``(P, M, K) int32`` key array of the centre genome,
            ``I_INF`` padded.
        digest_keys: ``(P, M) int64`` composites of the population, as produced by
            :func:`prepare_family` — ascending, padding last.
        key_span: Static key bound ``S`` from the genome.

    Returns:
        ``(P, M) int32`` row index **into the digest's sorted attribute block** per gene of
        the centre, ``I_INF`` where the population has no homologue and for the centre's
        padded rows.  This is the same table ``alignment.find_homologous`` returns, with
        digest-ordered instead of payload-ordered row numbers, so the caller reads the
        population attributes straight out of the digest.

    Raises:
        ConfigError: If the payload key width disagrees with the gene, or if the centre and
            digest key tables carry a different gene-axis length.
        evoxc.ShapeError: Propagated unchanged if the concatenation disagrees on shapes.

    Notes:
        One sort of the ``2 * M`` concatenation plus one adjacency comparison:
        because the composites are injective and the two sides carry different padding
        sentinels, ``right == left + 1`` holds exactly on ``(centre gene, population gene)``
        pairs with equal key rows, so no key-row gather and no ``reduce_min`` is needed.
        The matched population column is written into the centre's *own* row through a
        ``flat_scatter`` whose masked writes land in a trailing dump column.
    """
    table = _key_table("measure", gene, keys)
    pop_size, count = int(table.shape[0]), int(table.shape[1])
    if tuple(int(size) for size in digest_keys.shape) != (pop_size, count):
        raise ConfigError(
            f"measure: the centre's key table is {(pop_size, count)} but the digest's is "
            f"{tuple(digest_keys.shape)}; pass one digest row per centre row"
        )
    pack, valid = _pack(gene, table, key_span)
    composite = evoxc.select(
        valid,
        evoxc.add(evoxc.multiply(pack, _i64(1 << _TAG_BITS)), _i64(_TAG_CENTRE)),
        _i64(_PAD_CENTRE),
    )
    merged = evoxc.concatenate((composite, digest_keys), axis=1)
    order = evoxc.cast(evoxc.argsort(merged, axis=-1, stable=False), evoxc.int32)
    # The permutation is already in hand, so a flat gather of it produces the sorted block
    # for one cheap gather instead of a second comparison sort of the same data.
    values = flat_gather(merged, order)
    pair_count = 2 * count - 1
    left = evoxc.slice(values, (0, 0), (pop_size, pair_count))
    right = evoxc.slice(values, (0, 1), (pop_size, pair_count))
    homologous = evoxc.equal(right, evoxc.add(left, _i64(1)))
    centre_col = evoxc.slice(order, (0, 0), (pop_size, pair_count))
    matched_col = evoxc.subtract(
        evoxc.slice(order, (0, 1), (pop_size, pair_count)), _i32(count)
    )
    # One extra trailing column absorbs every dropped write: masked-out positions carry a
    # centre column outside ``[0, count)`` and a population column that may be negative.
    table_out = enp.full((pop_size, count + 1), I_INF, dtype=evoxc.int32)
    destination = evoxc.select(homologous, centre_col, _i32(count))
    scattered = flat_scatter(table_out, destination, matched_col)
    return evoxc.slice(scattered, (0, 0), (pop_size, count))


class DistanceDigest(NamedTuple):
    """Prepared population digest for both gene families, one generation of reuse.

    Flattened field-by-field (rather than nesting two :class:`FamilyDigest` tuples) so the
    value is an ordinary tuple of six tensors, exactly like
    :class:`~tensorneat.genome.utils.GenomeTensors`.

    Attributes:
        node_keys (evoxc.SymbolicTensor): ``(P, N) int64`` packed node-key composites.
        node_attrs (evoxc.SymbolicTensor): ``(P, N, NA) float32`` node attributes in the
            same order.
        node_count (evoxc.SymbolicTensor): ``(P,) int32`` real node genes per individual.
        conn_keys (evoxc.SymbolicTensor): ``(P, E) int64`` packed connection-key composites.
        conn_attrs (evoxc.SymbolicTensor): ``(P, E, CA) float32`` connection attributes in
            the same order.
        conn_count (evoxc.SymbolicTensor): ``(P,) int32`` real connection genes per
            individual.
    """

    node_keys: evoxc.SymbolicTensor
    node_attrs: evoxc.SymbolicTensor
    node_count: evoxc.SymbolicTensor
    conn_keys: evoxc.SymbolicTensor
    conn_attrs: evoxc.SymbolicTensor
    conn_count: evoxc.SymbolicTensor

    @property
    def field_names(self) -> tuple[str, ...]:
        """Field names in payload order, for generic digest walks.

        Returns:
            The six field names of the digest, node family first.
        """
        return self._fields


__all__ = ["DistanceDigest", "FamilyDigest", "match_rows", "prepare_family"]
