"""Dense key-to-gene table: dispatch-proportional distance for a single centre.
The row-wise merge in :mod:`.digest` is *sort proportional*: every ``measure`` call sorts the
``2 * M`` concatenation of the population digest and one centre genome, so a generation that
measures ``2 * species_size`` centres pays ``2 * species_size`` sorts of the same population.
When the centre is **one** genome that work is avoidable, because a genome is a sparse table
of ``(key, gene)`` rows and NEAT keys live in a small static range:
* node family — ``d = node_key``, table width ``S + 1``;
* connection family — ``d = in_key * S + out_key``, table width ``S * S + 1``;
with ``S = genome.key_span``.  Scattering the centre's rows into such a table and *gathering*
it at the population's own dense index block resolves every homologue in one pass: no sort, no
merge, no inverse permutation.
Two cells of each table are not genes.  The **dump** cell (``width - 1``) catches every centre
row whose key is absent or outside ``[0, S)``; the **padding** cell (``width``) is never owned
and is what a population gene with an unusable key reads.  Both start at the gene count — an
impossible row index — so ``owner_at < count`` is the whole homology test, and no out-of-range
index ever reaches the backend.
Precondition (the payload contract of :mod:`.digest`, restated because this route depends on it):
gene keys are non-negative ``int32`` values **below** ``genome.key_span``.  A key outside
``[0, S)`` is *unmatched* here — exactly like the execution schedule drops it
(:mod:`tensorneat.common.tensorization`) — while the row-wise merge would still match it against
the same out-of-range key in the other genome.  The two routes therefore agree on every payload
whose keys respect the bound, and the algorithm's structural-key allocator is what keeps them
there.  The divergence is not silent: :func:`.digest.prepare_family` counts the present genes
the table cannot address and :class:`.digest.DistanceDigest` carries the two per-individual
counts as ``node_oor``/``conn_oor``.  Nothing here raises or branches on them; a caller that
relies on this route asserts they are zero on the host (the same host-side pattern as
``pipeline.py``'s ``max_overflow`` check) or keeps
``DefaultDistance(dense_single_centre=False)``.  Two key rows with the same key tuple inside one
genome are outside the payload contract as well (``alignment`` documents the same); the dump
redirect makes them a last-writer-wins scatter rather than a crash.
Static gate, never a runtime branch: a family takes this route only while its tables fit
:data:`_DENSE_MAX_BYTES` (float32 bytes of the centre tables), and a pipeline whose *two*
families both qualify is the only one that uses it at all — a partially dense ``measure`` would
have to broadcast the centre payload for the other family, and the row-wise merge is then the
honest route.  ``S = 150`` (``max_nodes = 50``, ``max_conns = 100``) gives ``22 501`` connection
cells, i.e. ~0.2 MB; ``S = 2000`` would need ~72 MB and falls back to the merge.
Determinism: the attribute term is reduced **in the centre's own row order**, exactly like the
merge's ``reduce_sum`` over the centre's gene axis (the pair distance is scattered into
``(P, M + 1)`` at the centre row each population gene homologues to, the dump column is sliced
off, and the surviving ``(P, M)`` block is reduced over axis 1).  Same values in the same
positions, so a dense ``measure`` is *bit-identical* to ``execute_distance`` on the same centre
and population — the property the equivalence gate asserts, not merely closeness.
Cost shape (measured, numpy reference interpreter, pop 1000): the reference interpreter charges
per *traced op* rather than per element, so what a call spends is its op count.  Everything that
depends only on the population geometry — the two table bases, the centre row index, the flat
reorder offsets and buffer, and the rank-0 scalars of the tail — is therefore built once per
generation in :func:`.digest.prepare_family` and travels here inside
:class:`~.layout.DenseLayout`; a ``measure`` call left with nothing but value-dependent work is
what the op counts below are made of.
Never emit ``evoxc.gather(..., axis=1)`` on a rank-3 block: it measured 163 ms for one
``(1000, 50, 4)`` gather at pop 1000, against 82 µs for the same values gathered on axis 0 from
the (already transposed) table — the dense table exists so that every gather here is an axis-0
gather.
"""
from typing import TYPE_CHECKING, NamedTuple
import evoxc
import evoxc.numpy as enp
from .consts import ConstantPool
from .digest import _FOLDED_KEYS, _column, dense_rows, dense_width
from .layout import reorder
if TYPE_CHECKING:  # pragma: no cover - typing only
    from ...base import BaseGenome
    from .base import BaseDistance
#: Upper bound on the float32 bytes of one family's centre tables (owner + attributes +
#: trailing markers).  Above it the row-wise merge is the route: the merge's cost grows with
#: the gene axis ``M`` while a dense table grows with ``key_span``, so a large key span
#: (``S * S`` connection cells) is exactly where the sort stops being the expensive side.
_DENSE_MAX_BYTES = 8 * 1024 * 1024
def enabled(gene, genome: "BaseGenome") -> bool:
    """True when this gene family may use the dense table (a static decision).
    Args:
        gene: Node or connection gene declaring ``key_width`` and ``attr_width``.
        genome ("BaseGenome"): Genome owning ``key_span``.
    Returns:
        bool: ``True`` when the family's centre tables fit :data:`_DENSE_MAX_BYTES` and the
        table width stays inside the ``int32`` index range.
    Notes:
        The tables are one cell wider than :func:`~.digest.dense_width` (the never-owned padding
        cell), which the byte estimate includes.
    """
    width = dense_width(gene, int(genome.key_span)) + 1
    if width > (1 << 31) - 1:  # the dense index is an int32 gather/scatter index
        return False
    marker_columns = max(0, int(gene.key_width) - _FOLDED_KEYS)
    # owner (int32) + one float32 column per attribute + one int32 column per trailing marker
    bytes_per_cell = 4 * (1 + int(gene.attr_width) + marker_columns)
    return width * bytes_per_cell <= _DENSE_MAX_BYTES
class FamilyTerms(NamedTuple):
    """The three value-dependent scalars one gene family contributes to the scalar tail.

    Attributes:
        count (evoxc.SymbolicTensor): rank-0 ``int32`` real genes of the centre genome,
            ``cnt1`` of the tail.
        matched (evoxc.SymbolicTensor): ``(P,) int32`` homologous gene count ``h`` per
            population individual.
        attribute (evoxc.SymbolicTensor): ``(P,) float32`` attribute distance summed over the
            homologous genes in the centre's own row order, or the layout's ``(P,)`` zeros for
            a family with no attribute column.
    """

    count: evoxc.SymbolicTensor
    matched: evoxc.SymbolicTensor
    attribute: evoxc.SymbolicTensor


def measure_centre(
    pool: ConstantPool,
    state,
    genome: "BaseGenome",
    distance: "BaseDistance",
    payload: tuple,
    digest,
) -> evoxc.SymbolicTensor | None:
    """Distance of one centre genome against a prepared population digest.

    Args:
        pool (ConstantPool): Per-trace rank-0 constant pool.
        state: Host configuration container, forwarded to the gene hook.
        genome ("BaseGenome"): Genome owning ``node_gene``/``conn_gene`` and ``key_span``.
        distance ("BaseDistance"): The operator supplying the scalar tail
            (``_scalar_distance``); the dense route never re-states the formula.
        payload (tuple): The four centre arrays ``(node_keys, node_attrs, conn_keys,
            conn_attrs)`` at batch ``1``.
        digest: The population ``DistanceDigest`` from :meth:`BaseDistance.prepare`.

    Returns:
        ``(P,) float32`` distances, or ``None`` when either family's table does not fit the
        static gate — the caller then takes the row-wise merge (its only correct shape for a
        population-shaped centre).

    Notes:
        The two families' tail inputs are **stacked on a leading axis** and handed to
        :meth:`~.default.DefaultDistance._scalar_distance` in one call, so a ``measure`` pays
        for the scalar formula (eleven traced ops) once instead of once per family.  The
        elementwise ops of the tail are shape-agnostic, and the final combine is a
        ``reduce_sum`` over the two-row axis, which is exactly the ``add`` the pair-wise route
        performs: the accumulator starts at ``0`` and ``0 + x`` is ``x`` for every distance
        (they are non-negative, and ``+0.0 + -0.0`` is ``+0.0`` anyway), so the two families
        are combined in the same order with the same values.
    """
    node_keys, node_attrs, conn_keys, conn_attrs = payload
    if not (enabled(genome.node_gene, genome) and enabled(genome.conn_gene, genome)):
        return None
    pop_size = int(digest.node_count.shape[0])
    node = _family_terms(
        pool,
        state,
        genome,
        genome.node_gene,
        node_keys,
        node_attrs,
        digest.node_dense,
        digest.node_attrs,
        digest.node_count,
        digest.node_key_cols,
        digest.node_layout,
    )
    conn = _family_terms(
        pool,
        state,
        genome,
        genome.conn_gene,
        conn_keys,
        conn_attrs,
        digest.conn_dense,
        digest.conn_attrs,
        digest.conn_count,
        digest.conn_key_cols,
        digest.conn_layout,
    )
    counts = evoxc.concatenate(
        (
            evoxc.reshape(node.count, (1, 1)),
            evoxc.reshape(conn.count, (1, 1)),
        ),
        axis=0,
    )
    matched = _stack((node.matched, conn.matched), pop_size)
    attribute = _stack((node.attribute, conn.attribute), pop_size)
    stacked = distance._scalar_distance(
        counts,
        matched,
        digest.tail_counts,
        attribute,
        pool,
        consts=digest.node_layout.tail,
    )
    return evoxc.reduce_sum(stacked, axes=0)


def _stack(blocks: tuple, pop_size: int) -> evoxc.SymbolicTensor:
    """Stack one ``(P,)`` block per gene family on a leading axis.

    Args:
        blocks (tuple): Two ``(P,)`` tensors of the same dtype.
        pop_size (int): Static population batch ``P``.

    Returns:
        evoxc.SymbolicTensor: ``(2, P)`` — row ``0`` the node family, row ``1`` the connection
        family, the layout the stacked scalar tail operates on.
    """
    return evoxc.concatenate(
        tuple(evoxc.reshape(block, (1, pop_size)) for block in blocks), axis=0
    )


def _family_terms(
    pool: ConstantPool,
    state,
    genome: "BaseGenome",
    gene,
    keys: evoxc.SymbolicTensor,
    attrs: evoxc.SymbolicTensor,
    pop_dense: evoxc.SymbolicTensor,
    pop_attrs: evoxc.SymbolicTensor,
    pop_count: evoxc.SymbolicTensor,
    pop_key_cols: evoxc.SymbolicTensor,
    layout,
) -> FamilyTerms:
    """One gene family of :func:`measure_centre`, as the three scalars the tail consumes.

    Args:
        pool (ConstantPool): Per-trace rank-0 constant pool.
        state: Host configuration container, forwarded to ``gene.distance``.
        genome ("BaseGenome"): Genome owning ``key_span`` (table geometry).
        gene: Node or connection gene.
        keys (evoxc.SymbolicTensor): ``(1, M)`` or ``(1, M, K) int32`` centre key rows.
        attrs (evoxc.SymbolicTensor): ``(1, M, A) float32`` centre attribute rows.
        pop_dense (evoxc.SymbolicTensor): ``(P, M) int32`` digest slot per population gene, in
            the digest's own order, redirected to the padding cell where the key is unusable.
        pop_attrs (evoxc.SymbolicTensor): ``(P, M, A) float32`` population attributes in the
            digest's order.
        pop_count (evoxc.SymbolicTensor): ``(P,) int32`` population gene count of this family.
        pop_key_cols (evoxc.SymbolicTensor): ``(P, M, K) int32`` population key columns in the
            digest's order (read only for a gene with trailing marker columns).
        layout: The family's :class:`~.layout.DenseLayout`, built once per generation.

    Returns:
        FamilyTerms: the centre gene count, the homologue count and the summed attribute
        distance of this family.

    Notes:
        Per-call ops, once the layout is hoisted, are the value-dependent ones only: the centre
        slot (one read per key column), one scatter, one gather, one range compare, the
        attribute table pair plus ``gene.distance``, the reorder scatter, two reductions.
        The owner table is scattered with the bare centre row index: no table cell outside a
        centre gene's own slot is ever *read* (a population gene with an unusable key reads the
        never-owned padding cell, not the dump cell), so redirecting the absent rows would be an
        op that changes nothing.
    """
    count = int(keys.shape[1])
    pop_size = int(pop_count.shape[0])
    slot, _, present = dense_rows(gene, int(genome.key_span), keys, pool, flat=True)
    # One owner row index per table cell: a centre gene writes its own row index at its slot.
    # Absent and out-of-range rows land in the dump cell, which nothing reads.
    owner = evoxc.scatter(layout.owner, slot, layout.row_index, axis=0)
    owner_at = evoxc.gather(owner, pop_dense, axis=0)
    matched = evoxc.less(owner_at, layout.count_i32)
    # Column the attribute values are written back at: the centre row of the homologue, or the
    # dump column when there is none.  A slot-only match is enough while the key width fits the
    # composite (the composite *is* the key tuple), so the redirect is free there — the star slot
    # already sends an unusable population key to the never-owned padding cell, whose owner is
    # ``count``.  With a trailing marker column the slot identifies ``(in, out)`` only, so a
    # population gene that shares the endpoints but differs in the marker is *not* homologous and
    # must not write its attribute distance into that centre row: one ``select`` keeps the
    # attribute sum equal to the merge's (measured: without it ``OriginConn`` payloads diverge by
    # ``compatibility_weight * |Δweight| / max(count)`` per shared-but-unmatched edge).
    column = owner_at
    marker_columns = int(gene.key_width) - _FOLDED_KEYS
    if marker_columns > 0:
        markers = evoxc.scatter(
            enp.zeros(
                (dense_width(gene, int(genome.key_span)) + 1, marker_columns),
                dtype=evoxc.int32,
            ),
            slot,
            _trailing_columns(keys, count, marker_columns),
            axis=0,
        )
        marker_at = evoxc.gather(markers, pop_dense, axis=0)
        for index in range(marker_columns):
            matched = evoxc.logical_and(
                matched,
                evoxc.equal(
                    _column(marker_at, index),
                    _column(pop_key_cols, _FOLDED_KEYS + index),
                ),
            )
        column = evoxc.select(matched, owner_at, layout.count_i32)
    attribute_distance = layout.zeros
    attr_width = int(attrs.shape[-1])
    if attr_width:
        table = evoxc.scatter(
            layout.table,
            slot,
            evoxc.reshape(attrs, (count, attr_width)),
            axis=0,
        )
        centre_attrs = evoxc.gather(table, pop_dense, axis=0)
        pair = gene.distance(state, centre_attrs, pop_attrs)
        # ``pair`` holds the distance of each *population* gene against the centre row it
        # homologues to.  The scalar tail sums over the centre's rows, so the values are written
        # back into the centre's order (dump column for everything else) before the reduction —
        # that is what keeps the float summation order, and with it the result, identical to the
        # row-wise merge.
        attribute_distance = evoxc.reduce_sum(
            evoxc.slice(
                reorder(pair, column, layout, pop_size, count),
                (0, 0),
                (pop_size, count),
            ),
            axes=1,
        )
    return FamilyTerms(
        evoxc.reduce_sum(evoxc.cast(present, evoxc.int32), axes=0),
        evoxc.reduce_sum(evoxc.cast(matched, evoxc.int32), axes=1),
        attribute_distance,
    )


def _trailing_columns(
    keys: evoxc.SymbolicTensor, count: int, marker_columns: int
) -> evoxc.SymbolicTensor:
    """Read the key columns past the folded prefix as one ``(M, marker_columns)`` block.

    Args:
        keys: ``(1, M, K) int32`` centre key rows.
        count: Static gene count ``M``.
        marker_columns: Static number of trailing columns ``K - _FOLDED_KEYS``.

    Returns:
        ``(M, marker_columns) int32`` marker columns of the single centre genome.
    """
    columns = [
        evoxc.reshape(_column(keys, _FOLDED_KEYS + index), (count, 1))
        for index in range(marker_columns)
    ]
    if len(columns) == 1:
        return columns[0]
    return evoxc.concatenate(tuple(columns), axis=1)


__all__ = ["enabled", "measure_centre"]
