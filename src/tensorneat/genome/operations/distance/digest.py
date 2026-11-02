"""Two-phase compatibility distance: one population digest, one merge per centre.

NEAT speciation measures a single centre genome against the whole population twice per
generation (centre reservation and member assignment), so the *same* population payload is
re-sorted ``2 * species_size`` times per generation.
This module splits that work: :func:`prepare_family` sorts one gene family of the population
**once** per generation (packed key composites, the attribute rows permuted into the same
order, and the valid-gene count), and :func:`measure_alignment` merges one centre genome
against that digest with a single sort of the ``2 * M`` concatenation — no per-pair ``O(M^2)``
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

Determinism: both sorts of this module (the digest's in :func:`prepare_family` and the merge's
two in :func:`measure_alignment`) use ``stable=True``, which is a **performance** choice, not
a tie-semantics one: the digest half of the merge is already ascending, so numpy's stable kind
(timsort) detects a single run and the merge's two sorts drop from 2.49 to 0.57 ms per
``(1000, 200)`` row block at ``M = 100`` (1.43 → 0.28 ms for the merge sort, 1.05 → 0.29 ms
for its inverse permutation), and the digest's own sort gains 15–17 % because a payload is a
short real-gene run followed by a long padding run.
Stability also *pins* the tie order (input order), which is strictly better defined than a
comparison sort's.
Ties can only join equal composites — padding rows (whose attributes are ``NaN`` in both sides
and which never pass the homology test) and, in a degenerate payload, duplicate key rows,
which cannot pair across sides anyway — so results and the digest's eight leaves are
byte-identical either way (verified on evolved payloads through the fixed-input oracle).

Cost discipline: this module never calls ``enp.full`` directly — every rank-0 constant comes
from the :class:`~.consts.ConstantPool` threaded in by the caller — and every row permutation
is one hand-built flat gather (``reshape`` + ``add`` + ``gather``) instead of
``common.tools.flat_gather``, whose out-of-range escape checks are pure overhead for indices
this module has already clamped.
"""

from typing import NamedTuple

import evoxc
import evoxc.numpy as enp

from ....common import (
    I_INF,
    ConfigError,
    batch_offsets,
    masked_flat_gather,
)
from ..alignment import key_rows
from .consts import ConstantPool

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


class AlignedGenes(NamedTuple):
    """One centre family resolved against a digest: the merge's finished answer.

    Produced by :func:`measure_alignment` and consumed by
    :meth:`~tensorneat.genome.operations.distance.base.BaseDistance.distance_from_alignment`,
    which is why the mask and both counts travel with the row table instead of being
    recomputed from it: the merge already knows them, and a second ``not_equal``/``reduce_sum``
    per family per call was pure duplication.

    Attributes:
        rows (evoxc.SymbolicTensor): ``(P, M) int32`` digest row of the homologue of each
            centre gene, **redirected to row ``0``** where there is none — so the aligned
            attribute gather needs no range check, no mask and no ``NaN`` fill.
        homologous (evoxc.SymbolicTensor): ``(P, M) bool`` True where that row is a real
            homologue (and not a centre padding row).
        count (evoxc.SymbolicTensor): ``(P,) int64`` real genes of the centre genome — the
            ``reduce_sum`` of the merge's own validity mask.
        matched (evoxc.SymbolicTensor): ``(P,) int64`` number of homologous genes ``h``.
    """

    rows: evoxc.SymbolicTensor
    homologous: evoxc.SymbolicTensor
    count: evoxc.SymbolicTensor
    matched: evoxc.SymbolicTensor


def _key_width(name: str, gene, keys: evoxc.SymbolicTensor) -> int:
    """Validate one payload key array against the gene and return its static width.

    Args:
        name: Operator name used in the error message.
        gene: Node or connection gene declaring ``key_width``.
        keys: ``(P, M)`` (one-column node key) or ``(P, M, K)`` key array.

    Returns:
        int: the number of key columns, ``1`` for a rank-2 array.

    Raises:
        ConfigError: If the array is neither 2-D nor 3-D, or if its key width is not the
            gene's ``key_width`` — the digest is built and consumed with one gene family at a
            time, so a mismatched width is a configuration error rather than a silent
            reinterpretation.

    Notes:
        Static shapes only: the width is read off ``Shape`` and no graph op is emitted, so a
        one-column node key needs neither a reshape nor a slice to be folded.
    """
    rank = len(keys.shape)
    if rank == 2:
        width = 1
    elif rank == 3:
        width = int(keys.shape[-1])
    else:
        raise ConfigError(
            "gene key tables must be (P, M) or (P, M, K), "
            f"got shape {tuple(keys.shape)}"
        )
    if width != int(gene.key_width):
        raise ConfigError(
            f"{name}: {type(gene).__name__} declares {gene.key_width} key column(s) but the "
            f"payload key array carries {width}"
        )
    return width


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


def _pack(
    keys: evoxc.SymbolicTensor, pool: ConstantPool
) -> tuple[evoxc.SymbolicTensor, evoxc.SymbolicTensor]:
    """Fold the first key columns of one gene family into injective ``int64`` composites.

    Args:
        keys: ``(P, M)`` or ``(P, M, K) int32`` key array, ``I_INF`` padded.
        pool (ConstantPool): Per-trace rank-0 constant pool.

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
        composite, so a third column is verified row-for-row by :func:`measure_alignment`
        instead.
        A rank-2 key array is its own leading column, so a node key costs no slice and no
        reshape at all; the leading column is read once and reused for the mask and the first
        digit.
    """
    if len(keys.shape) == 2:
        valid = evoxc.not_equal(keys, pool.i32(I_INF))
        return evoxc.cast(keys, evoxc.int64), valid
    leading = _column(keys, 0)
    valid = evoxc.not_equal(leading, pool.i32(I_INF))
    composite = evoxc.cast(leading, evoxc.int64)
    width = int(keys.shape[2])
    for index in range(1, min(width, _FOLDED_KEYS)):
        composite = evoxc.add(
            evoxc.multiply(composite, pool.i64(_KEY_BASE)),
            evoxc.cast(_column(keys, index), evoxc.int64),
        )
    return composite, valid


def _tagged(
    pack: evoxc.SymbolicTensor,
    valid: evoxc.SymbolicTensor,
    tag: int,
    padding: int,
    pool: ConstantPool,
) -> evoxc.SymbolicTensor:
    """Attach the side tag to a folded composite, or the side's padding sentinel.

    Args:
        pack: ``(P, M) int64`` folded key value.
        valid: ``(P, M) bool`` real-gene mask.
        tag: ``_TAG_CENTRE`` or ``_TAG_POPULATION``.
        padding: The side's sentinel for absent rows.
        pool (ConstantPool): Per-trace rank-0 constant pool.

    Returns:
        ``(P, M) int64`` sortable composite.

    Notes:
        ``pack * 2 + tag`` keeps ``_TAG_BITS`` low bits for the side, and the multiplication
        cannot overflow: ``pack < 2**62`` and ``tag < 2`` keep the result below ``2**63``.
        The centre's tag is ``0``, so its tagged value is exactly ``pack * 2`` and the ``add``
        is skipped — the centre is tagged once per family per ``measure`` call and that
        addition would be dead arithmetic.
    """
    scaled = evoxc.multiply(pack, pool.i64(1 << _TAG_BITS))
    if tag:
        scaled = evoxc.add(scaled, pool.i64(tag))
    return evoxc.select(valid, scaled, pool.i64(padding))


def _gather_rows(
    rows: evoxc.SymbolicTensor,
    order: evoxc.SymbolicTensor,
    width: int,
    pool: ConstantPool,
) -> evoxc.SymbolicTensor:
    """Permute the rows of a ``(P, M, W)`` block into a gene order.

    Args:
        rows: ``(P, M, W)`` block (attribute rows or key columns).
        order: ``(P, M) int32`` permutation (row ``j`` of the result is row ``order[j]``).
        width: Static row width ``W``.
        pool (ConstantPool): Per-trace rank-0 constant pool.

    Returns:
        ``(P, M, W)`` permuted block.

    Notes:
        One flat gather with a ``row * W + column`` index block.  ``common.tools.flat_gather``
        would add its out-of-range escape checks on top, which are dead weight here: ``order``
        is a permutation of ``[0, M)``, so no index can escape.
    """
    pop_size, count = int(order.shape[0]), int(order.shape[1])
    flat = evoxc.reshape(rows, (pop_size * count * width,))
    offsets = enp.reshape(evoxc.multiply(order, pool.i32(width)), (pop_size, count, 1))
    columns = enp.arange(0, width, dtype=evoxc.int32)
    # Row stride as a ``(P, 1, 1)`` block: ``batch_offsets`` returns ``(P, 1)``, which would
    # broadcast against the last two axes of a 3-D index block instead of the leading one.
    rows_stride = enp.reshape(
        enp.arange(0, pop_size * count * width, count * width, dtype=evoxc.int32),
        (pop_size, 1, 1),
    )
    index = evoxc.add(evoxc.add(offsets, columns), rows_stride)
    return evoxc.reshape(evoxc.gather(flat, index, axis=0), (pop_size, count, width))


def _permute(
    block: evoxc.SymbolicTensor, order: evoxc.SymbolicTensor, pool: ConstantPool
) -> evoxc.SymbolicTensor:
    """Permute the columns of a ``(P, M)`` block into a gene order.

    Args:
        block: ``(P, M)`` block to permute.
        order: ``(P, M) int32`` permutation of ``[0, M)``.
        pool (ConstantPool): Per-trace rank-0 constant pool.

    Returns:
        ``(P, M)`` block whose column ``j`` is column ``order[j]`` of ``block``.

    Notes:
        The ``(P, M)`` twin of :func:`_gather_rows` (``W == 1``, so no column term and no
        result reshape).
    """
    pop_size, count = int(block.shape[0]), int(block.shape[1])
    flat = evoxc.reshape(block, (pop_size * count,))
    index = evoxc.add(order, batch_offsets(pop_size, count))
    return evoxc.gather(flat, index, axis=0)


def prepare_family(
    state, gene, keys: evoxc.SymbolicTensor, attrs: evoxc.SymbolicTensor, *, consts=None
) -> FamilyDigest:
    """Sort one gene family of a population into a reusable digest.

    Args:
        state: Host configuration container (unused; kept for the operator call shape).
        gene: Node or connection gene declaring the key and attribute layout.
        keys: ``(P, M)`` or ``(P, M, K) int32`` key array of the population, ``I_INF``
            padded.
        attrs: ``(P, M, A) float32`` attribute array of the population, NaN padded.
        consts (ConstantPool | None): Pool to draw rank-0 constants from; ``None`` starts a
            private one.  Pass one pool for a whole ``prepare``/``measure`` call (or a whole
            generation) so identical constants cost one op instead of one per use site.

    Returns:
        FamilyDigest: the packed composites (population tag) in ascending order with the
        padding sentinel last, the attribute rows and key columns permuted into that order,
        and the number of real genes per individual.

    Notes:
        This is the phase that used to run once per distance call; it now runs once per
        generation, so its cost is amortised over ``2 * species_size`` measurements.
        The sort is stable, which both pins the digest's tie order (input row order, where the
        previous comparison sort left it to numpy) and costs less: the payload's real genes
        are a run followed by a long ``NaN``/``I_INF`` padding run, so the interpreter's
        timsort path measured 1.51 → 1.25 ms (node) and 1.43 → 1.22 ms (conn) at ``P = 1000``.
        Ties are padding rows only (real keys are unique inside a genome), so the digest's
        eight leaves are byte-identical either way.
    """
    pool = consts if consts is not None else ConstantPool()
    width = _key_width("prepare", gene, keys)
    pack, valid = _pack(keys, pool)
    tagged = _tagged(pack, valid, _TAG_POPULATION, _PAD_POPULATION, pool)
    order = evoxc.cast(evoxc.argsort(tagged, axis=-1, stable=True), evoxc.int32)
    attrs_sorted = _gather_rows(attrs, order, int(attrs.shape[-1]), pool)
    # A digest key block is always rank 3: the unf folded case keeps the *unpermuted* table
    # (nothing reads it) but still in the documented ``(P, M, K)`` spelling.
    key_cols_sorted = (
        _gather_rows(keys, order, width, pool)
        if width > _FOLDED_KEYS
        else key_rows(keys)
    )
    count = evoxc.cast(evoxc.reduce_sum(valid, axes=1), evoxc.int32)
    return FamilyDigest(
        _permute(tagged, order, pool), attrs_sorted, count, key_cols_sorted
    )


def measure_alignment(
    gene,
    keys: evoxc.SymbolicTensor,
    digest_keys: evoxc.SymbolicTensor,
    digest_key_cols: evoxc.SymbolicTensor,
    *,
    consts=None,
) -> AlignedGenes:
    """Locate, for every gene of a centre genome, its homologue row in the digest.

    Args:
        gene: Node or connection gene declaring the key width.
        keys: ``(P, M)`` or ``(P, M, K) int32`` key array of the centre genome, ``I_INF``
            padded.
        digest_keys: ``(P, M) int64`` composites of the population, as produced by
            :func:`prepare_family` — ascending, padding last.
        digest_key_cols: ``(P, M, K) int32`` population key columns in the same order, the
            exact-match oracle for a gene whose key width exceeds ``_FOLDED_KEYS``.
        consts (ConstantPool | None): Pool to draw rank-0 constants from; ``None`` starts a
            private one.

    Returns:
        AlignedGenes: the digest row, the homology mask, the centre's gene count and its
        homologue count — everything the distance tail needs, so no consumer has to re-derive
        a mask or a count from the row table.

    Raises:
        ConfigError: If the payload key width disagrees with the gene, or if the centre and
            digest key tables carry a different gene-axis length.
        evoxc.ShapeError: Propagated unchanged if the concatenation disagrees on shapes.

    Notes:
        One sort of the ``2 * M`` concatenation, one ``argsort`` of its permutation (the
        inverse permutation, two ops — a scatter of the ranks cost eight and measured slower
        even before the sort became stable), then **two centre-width gathers** — the successor
        of each centre gene in the merged order and the digest row that successor carries.
        Reading the sorted block itself would be a ``2 * M``-wide gather of ``int64``
        composites (6.5 ms against 1.9 ms for everything else here at ``4P = 4000``); the
        inverse permutation is what lets every later step stay at the centre's own width ``M``.
        Both sorts are **stable**, which is a measured performance choice: the digest half of
        the concatenation is already ascending and the interpreter's stable ``argsort``
        (timsort) detects that run, so the merge's two sorts drop from 2.49 ms to 0.57 ms per
        ``(1000, 200)`` row block at ``M = 100`` (the whole conn-family aligner 4.89 → 2.43 ms)
        at an unchanged op count.  Ties only ever join equal composites — padding rows and, in
        a degenerate payload, duplicate keys — so nothing observable moves.
        The fold is injective on the folded columns and the two sides carry different side
        tags, so a homologous pair is exactly a centre gene whose *successor* in merged order
        is the population composite ``composite + 1`` — no key-row gather and no
        ``reduce_min`` — and the tag bit is checked explicitly, because
        ``(population, centre)`` composites can also differ by one (that is what the second
        tag bit used to rule out).
        The successor position is clamped to ``2 * M - 1`` instead of being range-checked and
        masked: the last merged position always holds a padded row (both sentinels sit above
        every real composite), and no sentinel can equal a real ``composite + 1``, so the
        clamped read cannot pass the homology test — one ``clamp`` in place of a ``less`` and
        a ``select``, with no separate "in range" mask to carry.
        A gene with more key columns than ``_FOLDED_KEYS`` verifies the whole key row of its
        candidate pair against ``digest_key_cols``, so its homology is full-tuple key-row
        equality, never a comparison truncated to the folded columns.
        The query positions are the centre's own gene rows ``0..M-1``, so no second scatter
        over the centre's row numbers and no full-size ``I_INF`` constant is needed to fill
        the result.
    """
    pool = consts if consts is not None else ConstantPool()
    width = _key_width("measure", gene, keys)
    pop_size, count = int(keys.shape[0]), int(keys.shape[1])
    if tuple(int(size) for size in digest_keys.shape) != (pop_size, count):
        raise ConfigError(
            f"measure: the centre's key table is {(pop_size, count)} but the digest's is "
            f"{tuple(digest_keys.shape)}; pass one digest row per centre row"
        )
    pack, valid = _pack(keys, pool)
    composite = _tagged(pack, valid, _TAG_CENTRE, _PAD_CENTRE, pool)
    merged_width = 2 * count
    # The digest side goes first so that the merged index of a population gene *is* its
    # digest row: the tail then reads the row table as ``successor`` and needs no
    # ``subtract(successor, count)`` (two ops and a constant per family per call).
    merged = evoxc.concatenate((digest_keys, composite), axis=1)
    merged_flat = evoxc.reshape(merged, (pop_size * merged_width,))
    # ``int32`` indices: ``gather`` accepts ``int64`` too, but the 64-bit sort below and the
    # 64-bit index blocks it would force cost ~10 % of this kernel's milliseconds for two
    # saved casts (measured on the reference backend at P = 1000 and 4000).
    order = evoxc.cast(evoxc.argsort(merged, axis=-1, stable=True), evoxc.int32)
    # ``order[i]`` is the merged index that sorts to position ``i``, so sorting the permutation
    # itself is its inverse: ``inverse[j]`` is the sorted position of merged index ``j`` — i.e.
    # of the centre gene that sits there (centre genes occupy merged columns ``count..2*count-1``).
    # ``order`` is a permutation, so stability cannot change the inverse; it is only the
    # cheaper sort on the reference interpreter (see the Notes).
    position = evoxc.slice(
        evoxc.cast(evoxc.argsort(order, axis=-1, stable=True), evoxc.int32),
        (0, count),
        (pop_size, count),
    )
    offsets = batch_offsets(pop_size, merged_width)
    # The last sorted position has no successor; clamping redirects that (necessarily
    # non-homologous, see the docstring) read to the largest composite instead of masking it.
    successor_position = evoxc.minimum(
        evoxc.add(position, pool.i32(1)), pool.i32(merged_width - 1)
    )
    successor = evoxc.gather(
        evoxc.reshape(order, (pop_size * merged_width,)),
        evoxc.add(successor_position, offsets),
        axis=0,
    )
    candidate = evoxc.gather(merged_flat, evoxc.add(successor, offsets), axis=0)
    # The successor of a centre composite is a homologue exactly when it is the population
    # composite of the same key — ``composite + 1``, one odd value.  A centre composite is
    # always even (its tag is 0, and so is the centre padding sentinel) and so is the
    # population padding sentinel, so equality with that odd value already proves the
    # candidate is a real population gene: no separate tag test is needed.
    homologous = evoxc.equal(candidate, evoxc.add(composite, pool.i64(1)))
    # The digest side occupies merged columns ``0..count-1``, so the merged index of a
    # population gene is its digest row: no offset arithmetic, one alias.
    matched_col = successor
    if width > _FOLDED_KEYS:
        homologous = evoxc.logical_and(
            homologous,
            _key_row_matches(
                keys,
                digest_key_cols,
                matched_col,
                homologous,
                width,
                pool,
            ),
        )
    rows = evoxc.select(homologous, matched_col, pool.i32(0))
    return AlignedGenes(
        rows,
        homologous,
        evoxc.reduce_sum(valid, axes=1),
        evoxc.reduce_sum(homologous, axes=1),
    )


def match_rows(
    gene,
    keys: evoxc.SymbolicTensor,
    digest_keys: evoxc.SymbolicTensor,
    digest_key_cols: evoxc.SymbolicTensor,
    *,
    consts=None,
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
        consts (ConstantPool | None): Pool to draw rank-0 constants from; ``None`` starts a
            private one.

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
        The merge itself is :func:`measure_alignment`; this entry point only re-spells its
        ``I_INF``-free row table as the documented ``I_INF``-padded one, for callers that
        hold nothing but rows.  The distance path skips that re-spelling and the mask/count
        re-derivation by consuming :class:`AlignedGenes` directly.
    """
    pool = consts if consts is not None else ConstantPool()
    aligned = measure_alignment(
        gene, keys, digest_keys, digest_key_cols, consts=pool
    )
    rows = evoxc.select(aligned.homologous, aligned.rows, pool.i32(I_INF))
    return evoxc.cast(rows, evoxc.int32)


def _key_row_matches(
    keys: evoxc.SymbolicTensor,
    digest_key_cols: evoxc.SymbolicTensor,
    pop_col: evoxc.SymbolicTensor,
    mask: evoxc.SymbolicTensor,
    width: int,
    pool: ConstantPool,
) -> evoxc.SymbolicTensor:
    """Compare the full key row of every centre gene against its candidate digest row.

    Args:
        keys: ``(P, M, K) int32`` centre key columns.
        digest_key_cols: ``(P, M, K) int32`` population key columns, digest order.
        pop_col: ``(P, M) int32`` digest row of every centre gene's candidate partner
            (``successor - count``); out of range where ``mask`` is false.
        mask: ``(P, M) bool`` candidate mask; every gather is masked with it, so the
            out-of-range rows above never reach the backend.
        width: Static key width ``K``.
        pool (ConstantPool): Per-trace rank-0 constant pool.

    Returns:
        ``(P, M) bool`` — True where every key column of the two rows is equal.

    Raises:
        evoxc.ShapeError: Propagated unchanged if a key width disagrees.

    Notes:
        Only called for a gene whose key width exceeds ``_FOLDED_KEYS`` (a
        ``historical_marker`` column does not fit the ``int64`` composite).
        The comparison is unrolled one column at a time against the *centre's* row (no
        centre-side gather at all — the query is already row-aligned), and the key axis is
        the innermost one, where ``reduce_min`` costs an order of magnitude more than the
        ``logical_and`` fold it would replace.
    """
    pop_size, count = int(keys.shape[0]), int(keys.shape[1])
    digest_flat = evoxc.reshape(digest_key_cols, (pop_size, count * width))
    safe_col = evoxc.select(mask, pop_col, pool.i32(0))
    offsets = evoxc.multiply(safe_col, pool.i32(width))
    same = None
    for index in range(width):
        candidate = masked_flat_gather(
            digest_flat,
            evoxc.add(offsets, pool.i32(index)),
            mask=mask,
            fill=0,
        )
        equal = evoxc.equal(_column(keys, index), candidate)
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


__all__ = [
    "AlignedGenes",
    "DistanceDigest",
    "FamilyDigest",
    "match_rows",
    "measure_alignment",
    "prepare_family",
]
