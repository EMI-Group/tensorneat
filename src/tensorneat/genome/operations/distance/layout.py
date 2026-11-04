"""Static geometry and per-generation statics of the two per-centre distance routes.

``measure`` runs once per centre genome — ``2 * species_size`` times per generation — and
every call needs the same *static* scaffolding, whichever route it takes:

* :class:`MeasureStatics` — the row-wise merge of :mod:`.digest` and its distance tail: the
  rank-0 scalars (``I_INF``, ``0``, ``0.0``, the key-fold base and tag scale, the two
  sentinels, the attribute width, the operator's three tail weights) plus the per-individual
  flat index blocks of the merge and of the aligned-attribute gather.
* :class:`DenseLayout` — the dense single-centre route of :mod:`.dense`: the owner table
  base, the centre attribute table base, the centre row index, the flat reorder offsets and
  the reorder buffer.

Those tensors depend only on ``(pop_size, gene count, attr width)``, so building them inside
``measure`` paid for them ``2 * species_size`` times — and, worse, each one is a *graph
constant*, so a generation graph carried one copy per call.

:func:`build_statics` and :func:`build` move that work into
:func:`.digest.prepare_family`, which runs once per generation: the tensors travel to every
``measure`` call as leaves of the digest and the calls spend their ops on value-dependent
work only.

The module also owns the *static geometry* those two bundles are built from, because it is the
same kind of derivation — Python ints read off static shapes, with no value-dependent op:

* :data:`_KEY_BASE` and :data:`_FOLDED_KEYS` — the radix and column count of the packed key
  identity both routes match genes by.
* :func:`_column` and :func:`_prefix` — the two static key-column reads of the two routes (a
  kept-axis slice, plus a ``reshape`` for the single column, because ``evoxc`` rejects an
  axis-dropping index).
* :func:`key_fold` — the ``(2, 1) int64`` weight matrix of the one-``dot`` two-column fold that
  :func:`.digest._pack_centre` applies to a connection key table.
* :func:`dense_width`, :func:`dense_rows` and :func:`dense_slot` — the key → slot geometry of
  the dense table, i.e. what :class:`DenseLayout` addresses and the static gate sizes.

Why the constants are *leaves* and not rebuilt per call: the reference interpreter charges
per traced op rather than per element (a ``(P, M)`` add and a rank-0 constant cost the same
order of magnitude at ``P = 1000``), so an op saved is the whole budget's unit — measured,
see the module notes in :mod:`.dense` and the cost table in ``CONTEXT.md``.

Slot geometry, restated because two of the scalars encode it: a family's dense table has
``dense_width`` cells (one per key tuple) plus **one** never-owned *padding* cell at index
``width``.  A centre row whose key is absent or outside ``[0, key_span)`` scatters into the
*dump* cell ``width - 1``; a population gene whose key is unusable reads the padding cell.
Both cells start at the gene count (an impossible row index), so ``owner_at < count`` is
exactly the homology test — the padding cell is what removes the ``& real`` term the round-1
version needed.
"""
from typing import NamedTuple
import evoxc
import evoxc.numpy as enp
from ....common import I_INF
from ...window import fill
from .consts import ConstantPool


#: Radix of the positional key fold.  Every legal gene key is a non-negative ``int32``, so
#: ``2**31`` makes the fold injective for the whole key range without a ``key_span`` bound.
_KEY_BASE = 1 << 31

#: Number of key columns folded into the composite.  The remaining ones (a
#: ``historical_marker``, i.e. 32 more bits) cannot fit below ``2**63`` and are verified
#: row-for-row against the digest's key block instead.
_FOLDED_KEYS = 2


def _column(table: evoxc.SymbolicTensor, index: int) -> evoxc.SymbolicTensor:
    """Read one static key column of a ``(..., K)`` table.

    Args:
        table: ``(P, M, K)`` key-column block, or the ``(M, K)`` block of one individual's
            gene rows (the fast path flattens its centre key table once).
        index: Static (non-negative) column index.

    Returns:
        ``table.shape[:-1]`` view of column ``index``.

    Notes:
        A kept-axis slice plus ``reshape`` because ``evoxc`` rejects an axis-dropping
        integer index and ``Ellipsis``/``None`` static indexing.
    """
    rank = len(table.shape)
    slicer = (slice(None),) * (rank - 1) + (slice(index, index + 1),)
    return evoxc.reshape(table[slicer], tuple(int(size) for size in table.shape[:-1]))


def _prefix(table: evoxc.SymbolicTensor, count: int) -> evoxc.SymbolicTensor:
    """Read the leading ``count`` key columns of a ``(..., K)`` table as a kept-axis block.

    Args:
        table: ``(P, M, K)`` key-column block, ``K >= count``.
        count: Static number of leading columns to keep.

    Returns:
        ``(P, M, count)`` view of the first ``count`` columns.

    Notes:
        One ``slice`` op and no ``reshape``: the column axis is kept, so the block is
        already the shape :func:`~.digest._pack_centre`'s ``dot`` contracts over.
    """
    rank = len(table.shape)
    slicer = (slice(None),) * (rank - 1) + (slice(0, int(count)),)
    return table[slicer]


def key_fold(
    pool: ConstantPool, *, key_base: int, tag_scale: int
) -> evoxc.SymbolicTensor:
    """Build the ``(2, 1) int64`` weight matrix of the two-column key fold.

    Args:
        pool (ConstantPool): Per-trace constant pool (the two rank-0 values it holds are
            shared with the rest of the family's scaffold).
        key_base (int): Radix of the positional fold (``2**31``).
        tag_scale (int): Side-tag scale of the composite (``2``).

    Returns:
        ``(2, 1) int64`` matrix ``[[key_base * tag_scale], [tag_scale]]`` — the fold *with*
        the tag scale, so ``dot(cast(keys, int64), fold)`` is the side-scaled composite in
        one op.

    Notes:
        ``evoxc.dot`` needs rank ``>= 2`` on both sides, so the weights are a ``(2, 1)``
        matrix rather than a ``(2,)`` vector.  The scale being folded into the weights is
        what makes an all-``I_INF`` row (an absent gene, whose every key column is the
        padding sentinel) fold to exactly the centre padding sentinel — see
        :data:`~.digest._PAD_CENTRE` — so the centre side needs no ``select``.
        Built once per generation into :attr:`MeasureStatics.fold_i64`; a ``measure`` call
        therefore traces neither of its two constants.
    """
    return evoxc.concatenate(
        (
            enp.full((1, 1), key_base * tag_scale, dtype=evoxc.int64),
            enp.full((1, 1), tag_scale, dtype=evoxc.int64),
        ),
        axis=0,
    )


class MeasureStatics(NamedTuple):
    """Per-generation scaffolding of one gene family's row-wise ``measure`` route.

    Every field is a graph constant or a static index table, so a ``measure`` call spends no
    op on any of them: they are built once in :func:`.digest.prepare_family` (one shared
    :class:`~.consts.ConstantPool` per generation, so a value both families need costs one op)
    and read as leaves of the digest.  Exactly one of these per family pair is what removed
    the ``2 * species_size`` copies of the same ten constants from a generation graph.

    Attributes:
        iinf_i32 (evoxc.SymbolicTensor): rank-0 ``int32`` ``I_INF``, the padding sentinel the
            key fold tests against.
        zero_i32 (evoxc.SymbolicTensor): rank-0 ``int32`` ``0`` — the row the merge redirects a
            gene without a homologue to.
        key_base_i64 (evoxc.SymbolicTensor): rank-0 ``int64`` ``2**31``, the radix of the
            positional key fold.
        tag_scale_i64 (evoxc.SymbolicTensor): rank-0 ``int64`` ``2`` — ``pack * 2`` leaves the
            low bit for the side tag.
        one_i64 (evoxc.SymbolicTensor): rank-0 ``int64`` ``1``, the population tag and the
            ``composite + 1`` homologue test.
        pad_centre_i64 (evoxc.SymbolicTensor): rank-0 ``int64`` padding sentinel of the centre
            side of the merge.
        fold_i64 (evoxc.SymbolicTensor): ``(2, 1) int64`` weight matrix of the two-column key
            fold (:func:`key_fold`) — the one ``dot`` that turns a ``(P, M, 2)`` key table into
            the centre side's scaled composite.  Read only by a family whose key row folds two
            columns; the one-column node family keeps its two-scalar spelling.
        pad_column_i64 (evoxc.SymbolicTensor): ``(P, 1) int64`` trailing column of the merged
            block, every entry the centre padding sentinel — above every real composite and
            every population sentinel, so it sorts last and every centre gene has a successor
            position (:func:`.digest.measure_alignment`).
        offsets (evoxc.SymbolicTensor): ``(P, 1) int32`` ``arange(0, P * (2M + 1), 2M + 1)`` —
            the row stride of the merged block (``batch_offsets``, hoisted).
        offsets_plus_one (evoxc.SymbolicTensor): ``(P, 1) int32`` ``offsets + 1``, so the
            successor position and its flat base are one ``add`` instead of two.
        attr (AttrStatics): the aligned-attribute scaffold the scalar tail consumes — shared
            with the pair-wise route, which passes its own through the same argument.
    """

    iinf_i32: evoxc.SymbolicTensor
    zero_i32: evoxc.SymbolicTensor
    key_base_i64: evoxc.SymbolicTensor
    tag_scale_i64: evoxc.SymbolicTensor
    one_i64: evoxc.SymbolicTensor
    pad_centre_i64: evoxc.SymbolicTensor
    fold_i64: evoxc.SymbolicTensor
    pad_column_i64: evoxc.SymbolicTensor
    offsets: evoxc.SymbolicTensor
    offsets_plus_one: evoxc.SymbolicTensor
    attr: "AttrStatics"


class AttrStatics(NamedTuple):
    """Per-generation scaffolding of one gene family's *attribute* distance term.

    Split out of :class:`MeasureStatics` because the family geometry it needs
    (``pop_size``, gene count, attribute width, the tail weights) is exactly what a caller
    without a digest knows: :meth:`~tensorneat.genome.operations.distance.base.BaseDistance.distance_from_alignment`
    accepts this bundle, so the row-wise route hands over the digest's copy while the pair-wise
    route builds the same four fields per call.

    Attributes:
        base (evoxc.SymbolicTensor): ``(P, 1) int32`` ``arange(0, P * M, M)`` — the row base of
            the aligned-attribute **row** gather, so one ``add`` against the ``(P, M)`` homologue
            table is the whole index; an empty ``(0, 1)`` block for a family with no attribute
            column.
        zero_f32 (evoxc.SymbolicTensor): rank-0 ``float32`` ``0.0``, the value a non-homologous
            gene of the attribute term is zeroed to.
        attrs_rows (evoxc.SymbolicTensor | None): the family's attribute block as a
            ``(P * M, A)`` **row table**, the gathered operand — hoisted for the same reason as
            :attr:`base`, since a ``reshape`` is a fully-charged op and the ``2 * species_size``
            calls of a generation used to pay it once each.  ``None`` for a caller that holds no
            attribute block (the width-``0`` scaffold), in which case the gather never runs.
        tail (tuple): the scalar tail's three rank-0 ``float32`` constants —
            ``compatibility_disjoint``, ``compatibility_weight`` and ``1.0`` — or ``()`` for an
            operator without a fixed tail (see
            :meth:`~tensorneat.genome.operations.distance.base.BaseDistance._tail_constants`).
    """

    base: evoxc.SymbolicTensor
    zero_f32: evoxc.SymbolicTensor
    attrs_rows: "evoxc.SymbolicTensor | None"
    tail: tuple


#: Placeholder index block of a family with no attribute column (``attr_width == 0``) and of
#: the standalone :func:`.digest.match_rows` entry, which knows no attribute width at all: a
#: ``0``-row index block, so the caller takes its static ``attr_width == 0`` branch.
_NO_ATTR_BASE = (0, 1)


def build_attr_statics(
    pool: ConstantPool,
    *,
    pop_size: int,
    count: int,
    attr_width: int,
    tail: tuple[float, float, float] | None,
    attrs: "evoxc.SymbolicTensor | None" = None,
) -> AttrStatics:
    """Build one family's :class:`AttrStatics` (once per generation).

    Args:
        pool (ConstantPool): Pool shared by both families of a ``prepare`` call.
        pop_size (int): Static population batch ``P``.
        count (int): Static gene count ``M`` of the family.
        attr_width (int): Static attribute width ``A``; ``0`` gives an empty base.
        tail (tuple[float, float, float] | None): ``(disjoint, weight, one)`` of the operator
            supplying the scalar tail, or ``None`` for an operator that has none.
        attrs (evoxc.SymbolicTensor | None): ``(P, M, A) float32`` attribute block whose
            ``(P * M, A)`` row view the aligned-attribute gather indexes; ``None`` keeps
            :attr:`AttrStatics.attrs_rows` unset (a caller with no attribute block).

    Returns:
        AttrStatics: the family's attribute-distance scaffold.

    Notes:
        ``base`` is a ``P``-scaled graph constant — ``(P, 1) int32``, ~2 KB at ``P = 500`` and
        ~4 KB at ``P = 1000`` — and it is exactly what the ``2 * species_size`` calls of a
        generation used to rebuild: a row-stride ``arange`` (one ``constant`` op) and the
        ``reshape`` in front of it (another).  The alternative, a *materialised* ``(P, M, A)``
        zero block, is what the population-scaled-fill rule in ``window.fill`` forbids: this one
        is a static index table, not a data buffer.
        The ``(P, M, A)`` index block the previous spelling built per call (``rows * A`` plus a
        ``(P, 1, A)`` base) cost 10-20× a contiguous arithmetic op on the reference interpreter
        (a two-axis broadcast ``add`` over 100 k int32 measured ~200 µs against ~9 µs for the
        ``(P, M) + (P, 1)`` form), which is why the gather is a **row** gather into the hoisted
        ``(P * M, A)`` view: one cheap ``add`` for the whole index and one ``np.take`` along
        axis 0 with ``A``-wide rows, which is itself 2-3× cheaper than the flat gather over the
        same element count.
    """
    return AttrStatics(
        (
            evoxc.reshape(
                enp.arange(0, pop_size * count, count, dtype=evoxc.int32), (pop_size, 1)
            )
            if attr_width
            else enp.zeros(_NO_ATTR_BASE, dtype=evoxc.int32)
        ),
        pool.f32(0.0),
        (
            evoxc.reshape(attrs, (pop_size * count, attr_width))
            if attrs is not None and attr_width
            else attrs
        ),
        (
            (pool.f32(tail[0]), pool.f32(tail[1]), pool.f32(1.0))
            if tail is not None
            else ()
        ),
    )


def build_statics(
    pool: ConstantPool,
    *,
    pop_size: int,
    count: int,
    attr_width: int,
    key_base: int,
    tag_scale: int,
    pad_centre: int,
    tail: tuple[float, float, float] | None,
    attrs: "evoxc.SymbolicTensor | None" = None,
) -> MeasureStatics:
    """Build one family's :class:`MeasureStatics` (once per generation).

    Args:
        pool (ConstantPool): Pool shared by both families of a ``prepare`` call, so a constant
            they have in common costs one op.
        pop_size (int): Static population batch ``P``.
        count (int): Static gene count ``M`` of the family.
        attr_width (int): Static attribute width ``A``; ``0`` gives an empty attribute base.
        key_base (int): Radix of the positional key fold (``2**31``).
        tag_scale (int): Side-tag scale of the composite (``2``).
        pad_centre (int): Padding sentinel of the centre side of the merge.
        tail (tuple[float, float, float] | None): ``(disjoint, weight, one)`` of the operator
            supplying the scalar tail, or ``None`` for an operator that has none.
        attrs (evoxc.SymbolicTensor | None): ``(P, M, A) float32`` attribute block of this
            family, hoisted into :attr:`AttrStatics.attrs_rows`; ``None`` when the caller has
            no block to hand over.

    Returns:
        MeasureStatics: the family's static scaffold.

    Notes:
        The merge's ``(P, 1)`` stride table is a ``P``-scaled graph constant — ~8 KB for both
        families at ``P = 1000`` — and it is exactly what the ``2 * species_size`` calls of a
        generation used to rebuild: an ``arange`` is one ``constant`` op and the ``reshape`` in
        front of it is another.  The derived stride costs one ``add`` *here* instead of one per
        call, and both rank-0 pools are the generation's own, so a value the other family also
        needs costs one op.
    """
    # The merged block carries the digest half, the centre half and one trailing column (see
    # :func:`.digest.measure_alignment`), so its row stride is ``2M + 1``.
    merged_width = 2 * count + 1
    offsets = evoxc.reshape(
        enp.arange(0, pop_size * merged_width, merged_width, dtype=evoxc.int32),
        (pop_size, 1),
    )
    return MeasureStatics(
        pool.i32(I_INF),
        pool.i32(0),
        pool.i64(key_base),
        pool.i64(tag_scale),
        pool.i64(1),
        pool.i64(pad_centre),
        key_fold(pool, key_base=key_base, tag_scale=tag_scale),
        fill((pop_size, 1), pad_centre, evoxc.int64),
        offsets,
        evoxc.add(offsets, pool.i32(1)),
        build_attr_statics(
            pool,
            pop_size=pop_size,
            count=count,
            attr_width=attr_width,
            tail=tail,
            attrs=attrs,
        ),
    )


class DenseLayout(NamedTuple):
    """Static scaffold of one gene family's dense single-centre route.

    Attributes:
        owner (evoxc.SymbolicTensor): ``(width + 1,) int32`` base of the owner table, every
            cell holding the gene count — the value an unowned cell keeps, so
            ``owner_at < count`` is the homology mask.
        table (evoxc.SymbolicTensor): ``(width + 1, A) float32`` base of the centre attribute
            table (zeros), empty ``(width + 1, 0)`` for a family with no attribute column.
        row_index (evoxc.SymbolicTensor): ``(M,) int32`` centre gene row index, the scatter's
            update block.
        offsets (evoxc.SymbolicTensor): ``(P, 1) int32`` ``arange(0, P * (M + 1), M + 1)`` —
            the row stride of the flat reorder scatter.
        base (evoxc.SymbolicTensor): ``(P * (M + 1),) float32`` zero base of the reorder
            buffer (the trailing column is the dump column and is never read).  Built with
            :func:`~tensorneat.genome.window.fill`, i.e. one element plus a broadcast: an
            ``enp.zeros`` of this shape would bake ``4 * P * (M + 1)`` bytes per family
            (608 000 B for both families at ``P = 1000``).
        count_i32 (evoxc.SymbolicTensor): rank-0 ``int32`` gene count ``M``.
        zeros (evoxc.SymbolicTensor): ``(P,) float32`` zeros — the attribute term of a family
            with no attribute column, so both families always contribute one and the shared
            scalar tail is traced once for the pair (see :mod:`.dense`); also a
            :func:`~tensorneat.genome.window.fill` rather than a population-sized constant.
        pad_i32 (evoxc.SymbolicTensor): rank-0 ``int32`` never-owned padding slot ``width``.
        dump_i32 (evoxc.SymbolicTensor): rank-0 ``int32`` dump slot ``width - 1``.
    """

    owner: evoxc.SymbolicTensor
    table: evoxc.SymbolicTensor
    row_index: evoxc.SymbolicTensor
    offsets: evoxc.SymbolicTensor
    base: evoxc.SymbolicTensor
    count_i32: evoxc.SymbolicTensor
    zeros: evoxc.SymbolicTensor
    pad_i32: evoxc.SymbolicTensor
    dump_i32: evoxc.SymbolicTensor


def build(
    pool: ConstantPool,
    *,
    pop_size: int,
    count: int,
    attr_width: int,
    width: int,
    dense: bool,
) -> DenseLayout:
    """Build one family's :class:`DenseLayout` (once per generation).

    Args:
        pool (ConstantPool): Pool to draw the rank-0 constants from; pass the pool a whole
            ``prepare`` call shares so identical values cost one op.
        pop_size (int): Static population batch ``P``.
        count (int): Static gene count ``M`` of the family.
        attr_width (int): Static attribute width ``A`` (``0`` for a gene with no attribute).
        width (int): Static dense table width, dump slot included (``dense_width``).  The key
            span itself is *not* a layout scalar: the range test lives in
            :func:`dense_rows` and reads ``key_span`` from the payload's genome.
        dense (bool): Static gate — ``False`` builds empty placeholders, because the route is
            unreachable for this family and the two big bases are exactly what the gate exists
            to avoid materialising.

    Returns:
        DenseLayout: the family's static scaffold.

    Notes:
        Only the scalars are unconditional.  The owner and attribute bases are constant
        tensors of ``width`` cells (``22 501`` connection cells at ``key_span = 150``, i.e.
        ~0.2 MB), which is why ``dense`` is a static argument and why the gate lives in
        :func:`.dense.enabled` rather than here.
    """
    count_i32 = pool.i32(count)
    pad_i32 = pool.i32(width)
    dump_i32 = pool.i32(width - 1)
    if not dense:
        empty = enp.zeros((0,), dtype=evoxc.float32)
        return DenseLayout(
            pool.i32(count),
            enp.zeros((0, 0), dtype=evoxc.float32),
            enp.zeros((0,), dtype=evoxc.int32),
            enp.zeros((0, 1), dtype=evoxc.int32),
            empty,
            count_i32,
            empty,
            pad_i32,
            dump_i32,
        )
    stride = count + 1
    return DenseLayout(
        enp.full((width + 1,), count, dtype=evoxc.int32),
        enp.zeros((width + 1, attr_width), dtype=evoxc.float32),
        enp.arange(0, count, dtype=evoxc.int32),
        evoxc.reshape(
            enp.arange(0, pop_size * stride, stride, dtype=evoxc.int32), (pop_size, 1)
        ),
        fill((pop_size * stride,), 0.0, evoxc.float32),
        count_i32,
        fill((pop_size,), 0.0, evoxc.float32),
        pad_i32,
        dump_i32,
    )


def reorder(values, column, layout: DenseLayout, pop_size: int, count: int):
    """Write each population gene's value into its centre row with one flat scatter.

    Args:
        values: ``(P, M) float32`` per-population-gene values.
        column: ``(P, M) int32`` centre row of each population gene, ``count`` where it has no
            usable homologue — so no mask or ``select`` is needed to redirect it.
        layout (DenseLayout): The family's static scaffold (offsets, base, scalars).
        pop_size (int): Static population batch ``P``.
        count (int): Static gene count ``M``.

    Returns:
        ``(P, M + 1) float32`` — column ``j`` holds the value of the population gene that
        homologues with centre row ``j``; the trailing (dump) column is never read.

    Notes:
        A raw flat ``evoxc.scatter``, not
        :func:`~tensorneat.common.tools.masked_flat_scatter`: every index is inside ``[0, M]``
        (the redirect happened in the caller), so the generic out-of-range escape checks would
        be pure overhead, and the offsets are hoisted into ``layout``.
        Duplicate targets are impossible — a centre row owns one slot, so at most one
        population gene per individual can name it.
    """
    stride = count + 1
    flat_index = evoxc.reshape(evoxc.add(layout.offsets, column), (-1,))
    scattered = evoxc.scatter(
        layout.base, flat_index, evoxc.reshape(values, (-1,)), axis=0
    )
    return evoxc.reshape(scattered, (pop_size, stride))


def dense_width(gene, key_span: int) -> int:
    """Static width of one gene family's dense key table, dump slot included.

    Args:
        gene: Node or connection gene declaring ``key_width``.
        key_span (int): Static key bound ``S`` of the payload's genome.

    Returns:
        int: ``S + 1`` for a one-column (node) key, ``S * S + 1`` for a two-or-more-column
        (connection) key — one slot per key tuple, plus the trailing dump slot.

    Notes:
        The slot is an ``int32`` gather/scatter index, so a width beyond ``2**31 - 1`` (a key
        span above ~46 000) is not addressable; the fast path's static gate checks that.

    Raises:
        ConfigError: If ``key_span`` is not in ``[0, 2**31)`` — the slot and the range test of
            :func:`dense_rows` are signed ``int32`` quantities.
    """
    span = int(key_span)
    if not 0 <= span < _KEY_BASE:
        raise ConfigError(
            f"dense key table: key_span must lie in [0, 2**31), got {span}"
        )
    return span + 1 if int(gene.key_width) == 1 else span * span + 1


def dense_rows(
    gene,
    key_span: int,
    keys: evoxc.SymbolicTensor,
    pool: ConstantPool,
    *,
    flat: bool = False,
) -> tuple[evoxc.SymbolicTensor, evoxc.SymbolicTensor, evoxc.SymbolicTensor]:
    """Map every gene row of one family to its dense slot, in-range flag and presence flag.

    Args:
        gene: Node or connection gene declaring ``key_width``.
        key_span (int): Static key bound ``S``; a key outside ``[0, S)`` is not addressable and
            lands in the dump slot.
        keys (evoxc.SymbolicTensor): ``(P, M)`` or ``(P, M, K) int32`` key rows, ``I_INF``
            padded.
        pool (ConstantPool): Per-trace rank-0 constant pool.
        flat (bool): Return the three blocks on the gene-row axis only (``(M,)``) instead of
            ``(P, M)`` — the centre side of the fast path is a single individual, and its slot is
            a scatter index of rank 1.

    Returns:
        ``(slot, real, present)`` — the table slot of each row (the dump slot ``D - 1`` for an
        absent or out-of-range key), the in-range mask and the presence mask
        (``key != I_INF``, the same predicate :func:`prepare_family` counts as ``cnt2``).

    Notes:
        Node keys are their own slot; a connection slot is ``in_key * S + out_key``, so the fold
        is injective for keys inside ``[0, S)`` and the trailing dump slot catches the rest.
        The range test is **signed and same-width** (``int32``): ``evoxc``'s ``iree`` backend
        declares no unsigned dtype, so the one-comparison ``cast(key, uint32) < S`` form cannot
        be exported.  ``0 <= key < S`` says the same thing for every ``int32`` key — including
        that a negative key lands in the dump slot exactly as ``I_INF`` does — as long as
        ``0 <= S < 2**31`` (guarded statically in :func:`dense_width`).
        One function for both callers because the leading key column feeds three of the results
        — the slot, the range test and the presence test — and reading it once is what keeps a
        centre family at one read per key column (the round-1 spelling read it twice).
        Trailing key columns (``OriginConn``'s ``historical_marker``) are *not* part of the slot
        and are *not* range tested: a marker is a population-wide identity matched exactly
        through its own table, not a key-space index.
        A connection slot multiplies a possibly sentinel ``int32`` by ``S`` before the select
        that discards it; the wrap is deterministic and never observed, and hoisting a
        ``select`` in front of the product would cost an op on every real row to protect it.
    """
    span = int(key_span)
    count = int(keys.shape[1])
    if flat and len(keys.shape) == 3:
        # One reshape for the whole table instead of one per key column: a centre key block is
        # a single individual, so its gene-row axis is all the caller reads.
        keys = evoxc.reshape(keys, (count, int(keys.shape[2])))
    dump = pool.i32(dense_width(gene, span) - 1)
    # Signed int32 bounds of the range test; ``bound`` is also the connection-fold radix.
    zero = pool.i32(0)
    bound = pool.i32(span)
    sentinel = pool.i32(I_INF)

    def _row(x):
        # Flattening the *operands* rather than the results keeps the three returned blocks on
        # one reshape apiece instead of three; a block-shaped caller skips it entirely.
        if not flat or len(x.shape) == 1:
            return x
        return evoxc.reshape(x, (count,))

    def _in_range(column):
        # ``0 <= column < span`` in signed int32: bit-identical to the unsigned form for every
        # int32 value while span < 2**31, and exportable to a backend without unsigned dtypes.
        return evoxc.logical_and(
            evoxc.greater_equal(column, zero), evoxc.less(column, bound)
        )

    if int(gene.key_width) == 1:
        column = keys if len(keys.shape) == 2 else _column(keys, 0)
        column = _row(column)
        real = _in_range(column)
        present = evoxc.not_equal(column, sentinel)
        return evoxc.select(real, column, dump), real, present
    first = _row(_column(keys, 0))
    second = _row(_column(keys, 1))
    real = evoxc.logical_and(_in_range(first), _in_range(second))
    slot = evoxc.add(evoxc.multiply(first, bound), second)
    present = evoxc.not_equal(first, sentinel)
    return evoxc.select(real, slot, dump), real, present


def dense_slot(
    gene, key_span: int, keys: evoxc.SymbolicTensor, pool: ConstantPool
) -> tuple[evoxc.SymbolicTensor, evoxc.SymbolicTensor]:
    """Map every gene row of one family to its dense key-table slot.

    Args:
        gene: Node or connection gene declaring ``key_width``.
        key_span (int): Static key bound ``S``; a key outside ``[0, S)`` is not addressable and
            lands in the dump slot.
        keys (evoxc.SymbolicTensor): ``(P, M)`` or ``(P, M, K) int32`` key rows, ``I_INF``
            padded.
        pool (ConstantPool): Per-trace rank-0 constant pool.

    Returns:
        ``(slot (P, M) int32, real (P, M) bool)`` — the table slot of each row (the dump slot
        ``D - 1`` for an absent or out-of-range key) and the row's real-gene mask.

    Notes:
        The documented two-value spelling of :func:`dense_rows`, kept for callers that want the
        slot and the in-range mask without the presence mask.
    """
    slot, real, _ = dense_rows(gene, key_span, keys, pool)
    return slot, real


__all__ = [
    "AttrStatics",
    "DenseLayout",
    "MeasureStatics",
    "build",
    "build_attr_statics",
    "build_statics",
    "dense_rows",
    "dense_slot",
    "dense_width",
    "key_fold",
    "reorder",
]
