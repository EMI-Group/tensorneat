"""Per-generation statics of the two per-centre distance routes.

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
        offsets (evoxc.SymbolicTensor): ``(P, 1) int32`` ``arange(0, P * 2M, 2M)`` — the row
            stride of the ``2M``-wide merged block (``batch_offsets``, hoisted).
        offsets_plus_one (evoxc.SymbolicTensor): ``(P, 1) int32`` ``offsets + 1``, so the
            successor position and its flat base are one ``add`` instead of two.
        offsets_last (evoxc.SymbolicTensor): ``(P, 1) int32`` ``offsets + (2M - 1)`` — the
            clamp target of the successor position, i.e. the last row of the *same*
            individual's merged block.
        attr (AttrStatics): the aligned-attribute scaffold the scalar tail consumes — shared
            with the pair-wise route, which passes its own through the same argument.
    """

    iinf_i32: evoxc.SymbolicTensor
    zero_i32: evoxc.SymbolicTensor
    key_base_i64: evoxc.SymbolicTensor
    tag_scale_i64: evoxc.SymbolicTensor
    one_i64: evoxc.SymbolicTensor
    pad_centre_i64: evoxc.SymbolicTensor
    offsets: evoxc.SymbolicTensor
    offsets_plus_one: evoxc.SymbolicTensor
    offsets_last: evoxc.SymbolicTensor
    attr: "AttrStatics"


class AttrStatics(NamedTuple):
    """Per-generation scaffolding of one gene family's *attribute* distance term.

    Split out of :class:`MeasureStatics` because the family geometry it needs
    (``pop_size``, gene count, attribute width, the tail weights) is exactly what a caller
    without a digest knows: :meth:`~tensorneat.genome.operations.distance.base.BaseDistance.distance_from_alignment`
    accepts this bundle, so the row-wise route hands over the digest's copy while the pair-wise
    route builds the same four fields per call.

    Attributes:
        width_i32 (evoxc.SymbolicTensor): rank-0 ``int32`` attribute width ``A``, the row stride
            of the aligned-attribute gather.
        base (evoxc.SymbolicTensor): ``(P, 1, A) int32`` ``columns + rows_stride``, the base of
            the aligned-attribute flat gather — one ``add`` per call on top of it; an empty
            ``(0, 1, 0)`` block for a family with no attribute column.
        zero_f32 (evoxc.SymbolicTensor): rank-0 ``float32`` ``0.0``, the value a non-homologous
            gene of the attribute term is zeroed to.
        tail (tuple): the scalar tail's three rank-0 ``float32`` constants —
            ``compatibility_disjoint``, ``compatibility_weight`` and ``1.0`` — or ``()`` for an
            operator without a fixed tail (see
            :meth:`~tensorneat.genome.operations.distance.base.BaseDistance._tail_constants`).
    """

    width_i32: evoxc.SymbolicTensor
    base: evoxc.SymbolicTensor
    zero_f32: evoxc.SymbolicTensor
    tail: tuple


#: Placeholder of a family with no attribute column (``attr_width == 0``) and of the
#: standalone :func:`.digest.match_rows` entry, which knows no attribute width at all: a
#: ``0``-width index block, so the caller takes its static ``attr_width == 0`` branch.
_NO_ATTR_BLOCK = (0, 1, 0)


def build_attr_statics(
    pool: ConstantPool,
    *,
    pop_size: int,
    count: int,
    attr_width: int,
    tail: tuple[float, float, float] | None,
) -> AttrStatics:
    """Build one family's :class:`AttrStatics` (once per generation).

    Args:
        pool (ConstantPool): Pool shared by both families of a ``prepare`` call.
        pop_size (int): Static population batch ``P``.
        count (int): Static gene count ``M`` of the family.
        attr_width (int): Static attribute width ``A``; ``0`` gives an empty base.
        tail (tuple[float, float, float] | None): ``(disjoint, weight, one)`` of the operator
            supplying the scalar tail, or ``None`` for an operator that has none.

    Returns:
        AttrStatics: the family's attribute-distance scaffold.

    Notes:
        ``base`` is a ``P``-scaled graph constant — ``(P, 1, A) int32``, ~8 KB at ``P = 500``
        and ~16 KB at ``P = 1000`` — and it is exactly what the ``2 * species_size`` calls of a
        generation used to rebuild: a ``columns`` ``arange`` (one ``constant`` op), a row-stride
        ``arange`` (another) and the ``reshape`` in front of it (a third).  The alternative, a
        *materialised* ``(P, M, A)`` zero block, is what the population-scaled-fill rule in
        ``window.fill`` forbids: this one is a static index table, not a data buffer.
    """
    return AttrStatics(
        pool.i32(attr_width),
        (
            evoxc.add(
                enp.arange(0, attr_width, dtype=evoxc.int32),
                evoxc.reshape(
                    enp.arange(
                        0,
                        pop_size * count * attr_width,
                        count * attr_width,
                        dtype=evoxc.int32,
                    ),
                    (pop_size, 1, 1),
                ),
            )
            if attr_width
            else enp.zeros(_NO_ATTR_BLOCK, dtype=evoxc.int32)
        ),
        pool.f32(0.0),
        (
            (pool.f32(tail[0]), pool.f32(tail[1]), pool.f32(1.0))
            if tail is not None
            else ()
        ),
    )


#: Placeholder of a family with no attribute column (``attr_width == 0``) and of the
#: standalone :func:`.digest.match_rows` entry, which knows no attribute width at all: a
#: ``0``-width index block, so the caller takes its static ``attr_width == 0`` branch.
_NO_ATTR_BLOCK = (0, 1, 0)


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

    Returns:
        MeasureStatics: the family's static scaffold.

    Notes:
        The merge's ``(P, 1)`` stride table is a ``P``-scaled graph constant — ~8 KB for both
        families at ``P = 1000`` — and it is exactly what the ``2 * species_size`` calls of a
        generation used to rebuild: an ``arange`` is one ``constant`` op and the ``reshape`` in
        front of it is another.  The two derived strides cost one ``add`` each *here* instead of
        one per call, and both rank-0 pools are the generation's own, so a value the other
        family also needs costs one op.
    """
    merged_width = 2 * count
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
        offsets,
        evoxc.add(offsets, pool.i32(1)),
        evoxc.add(offsets, pool.i32(merged_width - 1)),
        build_attr_statics(
            pool,
            pop_size=pop_size,
            count=count,
            attr_width=attr_width,
            tail=tail,
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
            :func:`.digest.dense_rows` and reads ``key_span`` from the payload's genome.
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


__all__ = [
    "AttrStatics",
    "DenseLayout",
    "MeasureStatics",
    "build",
    "build_attr_statics",
    "build_statics",
    "reorder",
]
