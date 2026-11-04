"""Per-generation statics of the dense single-centre route.

The dense route (:mod:`.dense`) is called once per centre genome — ``2 * species_size``
times per generation — and every call needs the same *static* scaffolding: the base of the
owner table, the base of the centre attribute table, the centre row index, the flat
reorder offsets, the reorder buffer, and a handful of rank-0 scalars (the gene count, the
populations's count as ``float32``, the padding slot and the dump slot).  Those tensors
depend only on ``(pop_size, gene count, attr width)``, so building them inside ``measure``
paid for them ``2 * species_size`` times — and, worse, each one is a *graph constant*, so a
generation graph carried one copy per call.

:func:`build` moves that work into :func:`.digest.prepare_family`, which runs once per
generation: the tensors travel to every ``measure`` call as leaves of
:class:`DenseLayout` and the calls spend their ops on value-dependent work only.

Why the constants are *leaves* and not rebuilt per call: the reference interpreter charges
per traced op rather than per element (a ``(P, M)`` add and a rank-0 constant cost the same
order of magnitude at ``P = 1000``), so an op saved is the whole budget's unit — measured,
see the module notes in :mod:`.dense`.

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
from ...window import fill
from .consts import ConstantPool


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
        tail (tuple): the scalar tail's three rank-0 ``float32`` constants —
            ``compatibility_disjoint``, ``compatibility_weight`` and ``1.0`` — or ``()`` when the
            route is gated off.  They are leaves because
            :meth:`~tensorneat.genome.operations.distance.default.DefaultDistance._scalar_distance`
            multiplies by them on every call, and ``enp.full((), v)`` is one traced op per
            call site (``evoxc`` never deduplicates constants).
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
    tail: tuple


def build(
    pool: ConstantPool,
    *,
    pop_size: int,
    count: int,
    attr_width: int,
    width: int,
    tail: tuple[float, float, float] | None,
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
        tail (tuple[float, float, float] | None): ``(disjoint, weight, one)`` of the operator
            supplying the scalar tail, or ``None`` for an operator that has none (its route then
            builds the constants per call, as the row-wise routes do).
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
    # Pooled, so the two families share one op per constant: same pool, same value, same tensor.
    tail_c = (
        (pool.f32(tail[0]), pool.f32(tail[1]), pool.f32(1.0))
        if (dense and tail is not None)
        else ()
    )
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
            tail_c,
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
        tail_c,
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


__all__ = ["DenseLayout", "build", "reorder"]
