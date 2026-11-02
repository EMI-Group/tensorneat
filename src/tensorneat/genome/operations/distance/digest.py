"""Two-phase compatibility distance: one population digest, one merge per centre.

NEAT speciation measures a single centre genome against the whole population twice per
generation (centre reservation and member assignment), so the *same* population payload is
re-sorted ``2 * species_size`` times per generation.
This module splits that work: :func:`prepare_family` sorts one gene family of the population
**once** per generation (packed key composites, the attribute rows permuted into the same
order, and the valid-gene count), and :func:`match_rows` merges one centre genome against
that digest with a single sort of the ``2 * M`` concatenation — no per-pair ``O(M^2)``
matrix, no re-sort of the population and no key-row re-verification for a one- or two-column
gene.

The composite is what makes the merge exact: a gene row folds its key columns into one
``int64`` value ``pack``, positionally and most significant first, in base ``2**31``.
That base is the *whole* non-negative ``int32`` key range, so the fold is injective for every
key a payload can legally hold and never consults ``key_span``: a key outside ``[0, key_span)``
(a payload the execution schedule would silently ignore, which ``algorithm/`` can produce
because it allocates structural keys per individual) still packs to its own composite instead
of aliasing another gene's.  Composite equality is therefore key-row equality, and the legacy
hash-then-verify pair (two key-row gathers plus a ``reduce_min``) disappears for a node key
and for an ``(in_key, out_key)`` connection key.

Two folded columns are all an ``int64`` composite can hold: the fold needs ``2 * 31 = 62``
bits and the low bit carries the side tag.  A gene with more key columns (``OriginConn``'s
``historical_marker``) therefore folds its first two columns and **verifies** the remaining
columns row-for-row against the digest's sorted key block, which keeps homology equal to
full-tuple key-row equality rather than to a truncated comparison.

The low bit of the sort key is the side tag — population ``1``, centre ``0`` — so a centre
gene sorts before its population counterpart and a homologous pair is an adjacent
``(centre, population)`` pair whose composites differ by one.  Both sides' padding rows carry
their own sentinel above every real composite, and the sentinels are two apart and even, so
neither a ``(padding, padding)`` nor a ``(padding, gene)`` adjacency can ever pass the
"differ by one and the right neighbour is population-tagged" test.

Determinism: the sort is not required to be stable (ties are padding rows and, within one
side, genes with equal key rows, which cannot pair across sides), so the cheap comparison
sort is used; the emitted distances do not depend on the sort order because the per-gene
terms are scattered back to the centre's own row order before the reduction (see
:meth:`tensorneat.genome.operations.distance.base.BaseDistance.gene_measure`).
"""

from typing import NamedTuple

import evoxc
import evoxc.numpy as enp

from ....common import I_INF, ConfigError, flat_gather, flat_scatter, masked_flat_gather
from ..alignment import key_rows

#: Width of the side tag carried in the low bits of the composite sort key.
_TAG_BITS = 1

#: Side tag of a centre gene / of a population gene.  A homologous pair differs by exactly
#: ``_TAG_POPULATION - _TAG_CENTRE`` in the composite, and the parity of the low bit is also
#: what tells the merge which side a sorted neighbour came from.
_TAG_CENTRE = 0
_TAG_POPULATION = 1

#: Radix of the positional key fold.  Every legal gene key is a non-negative ``int32``, so
#: ``2**31`` makes the fold injective for the whole key range without a ``key_span`` bound.
_KEY_BASE = 1 << 31

#: Number of key columns folded into the composite.  The remaining ones (a
#: ``historical_marker``, i.e. 32 more bits) cannot fit below ``2**63`` and are verified
#: row-for-row against the digest's key block instead.
_FOLDED_KEYS = 2

#: Padding sentinels of the two sides: above every real composite, even (so they never pass
#: the "the right neighbour is population-tagged" test) and two apart (so a
#: ``(centre padding, population padding)`` adjacency never differs by one).
_PAD_CENTRE = (1 << 63) - 2
_PAD_POPULATION = (1 << 63) - 4


class FamilyDigest(NamedTuple):
    """Prepared form of one gene family of a population (batch-first, static shapes).

    Attributes:
        keys (evoxc.SymbolicTensor): ``(P, M) int64`` packed key composites, ascending with
            the padding sentinel last.
        attrs (evoxc.SymbolicTensor): ``(P, M, A) float32`` attribute rows permuted into the
            order of :attr:`keys`; absent rows keep their ``NaN`` attributes.
        count (evoxc.SymbolicTensor): ``(P,) int32`` number of real genes per individual, i.e.
            the length of the valid prefix of :attr:`keys`.
        key_cols (evoxc.SymbolicTensor): ``(P, M, K) int32`` key columns permuted into the
            same order — the exact-match oracle for the columns the composite does not fold.
            Read only when ``K > _FOLDED_KEYS``; for a narrower key the fold is already
            injective, so this field keeps the *unpermuted* key table and nothing reads it.
    """

    keys: evoxc.SymbolicTensor
    attrs: evoxc.SymbolicTensor
    count: evoxc.SymbolicTensor
    key_cols: evoxc.SymbolicTensor


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


def _pack(
    table: evoxc.SymbolicTensor,
) -> tuple[evoxc.SymbolicTensor, evoxc.SymbolicTensor]:
    """Fold the first key columns of one gene family into injective ``int64`` composites.

    Args:
        table: ``(P, M, K) int32`` key-column block, ``I_INF`` padded.

    Returns:
        ``(pack (P, M) int64, valid (P, M) bool)`` — the folded key value and the row's
        "real gene" mask (the leading key column is not ``I_INF``).

    Notes:
        Key columns are folded positionally, most significant first, in base ``_KEY_BASE``
        — one ``multiply``/``add`` pair per folded column, because an innermost-axis integer
        ``reduce_sum`` over the key axis costs an order of magnitude more than those
        elementwise ops, and the fold is exact either way because the arithmetic is integer.
        The base is the full non-negative ``int32`` range, so the fold is injective for every
        legal key whether or not it lies inside ``[0, key_span)``; a key beyond ``key_span``
        (which the schedule ignores) can therefore never alias another gene here.
        No digit is gated: ``2**31`` already bounds every legal digit, so no fold can
        overflow, and an absent row's digits are irrelevant because the caller replaces its
        whole composite with the side's sentinel.  (Gene keys are non-negative ``int32`` by
        contract — the execution schedule treats a negative key as absent.)
        Only ``_FOLDED_KEYS`` columns are folded: ``2 * 31`` bits already fill the ``int64``
        composite, so a third column is verified row-for-row by :func:`match_rows` instead.
    """
    leading = _column(table, 0)
    valid = evoxc.not_equal(leading, _i32(I_INF))
    width = int(table.shape[2])
    composite = None
    for index in range(min(width, _FOLDED_KEYS)):
        digit = evoxc.cast(_column(table, index), evoxc.int64)
        if composite is None:
            composite = digit
            continue
        composite = evoxc.add(evoxc.multiply(composite, _i64(_KEY_BASE)), digit)
    return composite, valid


def _tagged(
    pack: evoxc.SymbolicTensor,
    valid: evoxc.SymbolicTensor,
    tag: int,
    padding: int,
) -> evoxc.SymbolicTensor:
    """Attach the side tag to a folded composite, or the side's padding sentinel.

    Args:
        pack: ``(P, M) int64`` folded key value.
        valid: ``(P, M) bool`` real-gene mask.
        tag: ``_TAG_CENTRE`` or ``_TAG_POPULATION``.
        padding: The side's sentinel for absent rows.

    Returns:
        ``(P, M) int64`` sortable composite.

    Notes:
        ``pack * 2 + tag`` keeps ``_TAG_BITS`` low bits for the side, and the multiplication
        cannot overflow: ``pack < 2**62`` and ``tag < 2`` keep the result below ``2**63``.
    """
    return evoxc.select(
        valid,
        evoxc.add(evoxc.multiply(pack, _i64(1 << _TAG_BITS)), _i64(tag)),
        _i64(padding),
    )


def _gather_rows(
    rows: evoxc.SymbolicTensor, order: evoxc.SymbolicTensor, width: int
) -> evoxc.SymbolicTensor:
    """Permute the rows of a ``(P, M, W)`` block into a gene order.

    Args:
        rows: ``(P, M, W)`` block (attribute rows or key columns).
        order: ``(P, M) int32`` permutation (row ``j`` of the result is row ``order[j]``).
        width: Static row width ``W``.

    Returns:
        ``(P, M, W)`` permuted block.
    """
    pop_size, count = int(order.shape[0]), int(order.shape[1])
    flat = evoxc.reshape(rows, (pop_size, count * width))
    offsets = enp.reshape(
        evoxc.multiply(order, _i32(width)), (pop_size, count, 1)
    )
    columns = enp.reshape(enp.arange(0, width, dtype=evoxc.int32), (1, 1, width))
    return flat_gather(flat, evoxc.add(offsets, columns))


def prepare_family(
    state, gene, keys: evoxc.SymbolicTensor, attrs: evoxc.SymbolicTensor
) -> FamilyDigest:
    """Sort one gene family of a population into a reusable digest.

    Args:
        state: Host configuration container (unused; kept for the operator call shape).
        gene: Node or connection gene declaring the key and attribute layout.
        keys: ``(P, M)`` or ``(P, M, K) int32`` key array of the population, ``I_INF``
            padded.
        attrs: ``(P, M, A) float32`` attribute array of the population, NaN padded.

    Returns:
        FamilyDigest: the packed composites (population tag) in ascending order with the
        padding sentinel last, the attribute rows and key columns permuted into that order,
        and the number of real genes per individual.

    Notes:
        This is the phase that used to run once per distance call; it now runs once per
        generation, so its cost is amortised over ``2 * species_size`` measurements.
        The sort is *not* stable on purpose: ties are padding rows only (real keys are
        unique inside a genome), and the distances do not depend on the tie order.
    """
    table = _key_table("prepare", gene, keys)
    pack, valid = _pack(table)
    tagged = _tagged(pack, valid, _TAG_POPULATION, _PAD_POPULATION)
    order = evoxc.cast(evoxc.argsort(tagged, axis=-1, stable=False), evoxc.int32)
    keys_sorted = flat_gather(tagged, order)
    attrs_sorted = _gather_rows(attrs, order, int(attrs.shape[-1]))
    key_cols_sorted = (
        _gather_rows(table, order, int(table.shape[-1]))
        if int(table.shape[-1]) > _FOLDED_KEYS
        else table
    )
    count = evoxc.cast(
        evoxc.reduce_sum(evoxc.cast(valid, evoxc.int32), axes=1), evoxc.int32
    )
    return FamilyDigest(keys_sorted, attrs_sorted, count, key_cols_sorted)


def match_rows(
    gene,
    keys: evoxc.SymbolicTensor,
    digest_keys: evoxc.SymbolicTensor,
    digest_key_cols: evoxc.SymbolicTensor,
) -> evoxc.SymbolicTensor:
    """Locate, for every gene of a centre genome, its homologue row in the digest.

    Args:
        gene: Node or connection gene declaring the key width.
        keys: ``(P, M)`` or ``(P, M, K) int32`` key array of the centre genome,
            ``I_INF`` padded.
        digest_keys: ``(P, M) int64`` composites of the population, as produced by
            :func:`prepare_family` — ascending, padding last.
        digest_key_cols: ``(P, M, K) int32`` population key columns in the same order, the
            exact-match oracle for a gene whose key width exceeds ``_FOLDED_KEYS``.

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
        One sort of the ``2 * M`` concatenation, one flat gather of its permutation (the
        sorted block for free, instead of a second comparison sort) and one adjacency
        comparison.
        The fold is injective on the folded columns and the two sides carry different side
        tags, so a homologous pair is exactly an adjacent ``(centre, population)`` pair whose
        composites differ by one — no key-row gather and no ``reduce_min`` — and the tag bit
        is checked explicitly, because ``(population, centre)`` composites can also differ by
        one (that is what the second tag bit used to rule out).
        A gene with more key columns than ``_FOLDED_KEYS`` verifies the whole key row of its
        candidate pair against ``digest_key_cols``, so its homology is full-tuple key-row
        equality, never a comparison truncated to the folded columns.
        The matched population column is written into the centre's *own* row through a
        ``flat_scatter`` whose masked writes land in a trailing dump column.
    """
    table = _key_table("measure", gene, keys)
    pop_size, count = int(table.shape[0]), int(table.shape[1])
    width = int(table.shape[2])
    if tuple(int(size) for size in digest_keys.shape) != (pop_size, count):
        raise ConfigError(
            f"measure: the centre's key table is {(pop_size, count)} but the digest's is "
            f"{tuple(digest_keys.shape)}; pass one digest row per centre row"
        )
    pack, valid = _pack(table)
    composite = _tagged(pack, valid, _TAG_CENTRE, _PAD_CENTRE)
    merged = evoxc.concatenate((composite, digest_keys), axis=1)
    order = evoxc.cast(evoxc.argsort(merged, axis=-1, stable=False), evoxc.int32)
    # The permutation is already in hand, so a flat gather of it produces the sorted block
    # for one cheap gather instead of a second comparison sort of the same data.
    values = flat_gather(merged, order)
    pair_count = 2 * count - 1
    left = evoxc.slice(values, (0, 0), (pop_size, pair_count))
    right = evoxc.slice(values, (0, 1), (pop_size, pair_count))
    adjacent = evoxc.equal(right, evoxc.add(left, _i64(1)))
    population_tagged = evoxc.equal(
        evoxc.bitwise_and(right, _i64(1)), _i64(_TAG_POPULATION)
    )
    homologous = evoxc.logical_and(adjacent, population_tagged)
    centre_col = evoxc.slice(order, (0, 0), (pop_size, pair_count))
    matched_col = evoxc.subtract(
        evoxc.slice(order, (0, 1), (pop_size, pair_count)), _i32(count)
    )
    if width > _FOLDED_KEYS:
        homologous = evoxc.logical_and(
            homologous,
            _key_rows_equal(
                table, digest_key_cols, centre_col, matched_col, homologous, width
            ),
        )
    # One extra trailing column absorbs every dropped write: masked-out positions carry a
    # centre column outside ``[0, count)`` and a population column that may be negative.
    table_out = enp.full((pop_size, count + 1), I_INF, dtype=evoxc.int32)
    destination = evoxc.select(homologous, centre_col, _i32(count))
    scattered = flat_scatter(table_out, destination, matched_col)
    return evoxc.slice(scattered, (0, 0), (pop_size, count))


def _key_rows_equal(
    table: evoxc.SymbolicTensor,
    digest_key_cols: evoxc.SymbolicTensor,
    centre_col: evoxc.SymbolicTensor,
    pop_col: evoxc.SymbolicTensor,
    mask: evoxc.SymbolicTensor,
    width: int,
) -> evoxc.SymbolicTensor:
    """Compare the full key rows of candidate ``(centre, population)`` pairs.

    Args:
        table: ``(P, M, K) int32`` centre key columns.
        digest_key_cols: ``(P, M, K) int32`` population key columns, digest order.
        centre_col: ``(P, L) int32`` centre row of every candidate, out of range where the
            candidate is masked out.
        pop_col: ``(P, L) int32`` population row of every candidate, likewise.
        mask: ``(P, L) bool`` candidate mask; every gather is masked with it, so the
            out-of-range rows above never reach the backend.
        width: Static key width ``K``.

    Returns:
        ``(P, L) bool`` — True where every key column of the two rows is equal.

    Notes:
        Only called for a gene whose key width exceeds ``_FOLDED_KEYS`` (a
        ``historical_marker`` column does not fit the ``int64`` composite).
        The comparison is unrolled one column at a time — the key axis is the innermost one,
        and an innermost-axis ``reduce_min`` over it costs an order of magnitude more than
        the ``logical_and`` fold it would replace.
    """
    pop_size, count = int(table.shape[0]), int(table.shape[1])
    centre_flat = evoxc.reshape(table, (pop_size, count * width))
    pop_flat = evoxc.reshape(digest_key_cols, (pop_size, count * width))
    same = None
    for index in range(width):
        slot = _i32(index)
        centre_rows = masked_flat_gather(
            centre_flat,
            evoxc.add(evoxc.multiply(centre_col, _i32(width)), slot),
            mask=mask,
            fill=0,
        )
        pop_rows = masked_flat_gather(
            pop_flat,
            evoxc.add(evoxc.multiply(pop_col, _i32(width)), slot),
            mask=mask,
            fill=0,
        )
        equal = evoxc.equal(centre_rows, pop_rows)
        same = equal if same is None else evoxc.logical_and(same, equal)
    return same


class DistanceDigest(NamedTuple):
    """Prepared population digest for both gene families, one generation of reuse.

    Flattened field-by-field (rather than nesting two :class:`FamilyDigest` tuples) so the
    value is an ordinary tuple of eight tensors, the four payload arrays plus a count and a
    sorted key block per family.

    Attributes:
        node_keys (evoxc.SymbolicTensor): ``(P, N) int64`` packed node-key composites.
        node_attrs (evoxc.SymbolicTensor): ``(P, N, NA) float32`` node attributes in the
            same order.
        node_count (evoxc.SymbolicTensor): ``(P,) int32`` real node genes per individual.
        node_key_cols (evoxc.SymbolicTensor): ``(P, N, NK) int32`` node key columns in that
            order (the verification oracle; a one-column node key never needs it).
        conn_keys (evoxc.SymbolicTensor): ``(P, E) int64`` packed connection-key composites.
        conn_attrs (evoxc.SymbolicTensor): ``(P, E, CA) float32`` connection attributes in
            the same order.
        conn_count (evoxc.SymbolicTensor): ``(P,) int32`` real connection genes per
            individual.
        conn_key_cols (evoxc.SymbolicTensor): ``(P, E, CK) int32`` connection key columns in
            that order, the oracle for a ``historical_marker`` gene.
    """

    node_keys: evoxc.SymbolicTensor
    node_attrs: evoxc.SymbolicTensor
    node_count: evoxc.SymbolicTensor
    node_key_cols: evoxc.SymbolicTensor
    conn_keys: evoxc.SymbolicTensor
    conn_attrs: evoxc.SymbolicTensor
    conn_count: evoxc.SymbolicTensor
    conn_key_cols: evoxc.SymbolicTensor

    @property
    def field_names(self) -> tuple[str, ...]:
        """Field names in payload order, for generic digest walks.

        Returns:
            The eight field names of the digest, node family first.
        """
        return self._fields


__all__ = ["DistanceDigest", "FamilyDigest", "match_rows", "prepare_family"]
