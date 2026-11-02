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
merge, no inverse permutation.  The per-``measure`` op count drops from the merge's ~150 to the
~30 this module needs, and every one of those ops is a single gather/scatter/select over the
population's ``(P, M)`` gene table — so the cost scales with the *population width* and the
*dispatch*, never with a sort.

The trailing slot of each table is a **dump slot** (index ``D - 1``).  A centre row whose key is
absent or outside ``[0, S)`` scatters there, and a population row whose key is outside ``[0, S)``
reads there, so no out-of-range index ever reaches the backend and no absent gene can match a
present one.

Precondition (the payload contract of :mod:`.digest`, restated because this route depends on it):
gene keys are non-negative ``int32`` values **below** ``genome.key_span``.  A key outside
``[0, S)`` is *unmatched* here — exactly like the execution schedule drops it
(:mod:`tensorneat.common.tensorization`) — while the row-wise merge would still match it against
the same out-of-range key in the other genome.  The two routes therefore agree on every payload
whose keys respect the bound, and the algorithm's structural-key allocator is what keeps them
there.  Two key rows with the same key tuple inside one genome are outside the payload contract
as well (``alignment`` documents the same); the dump redirect makes them a last-writer-wins
scatter rather than a crash.

Static gate, never a runtime branch: a family takes this route only while its table fits
:data:`_DENSE_MAX_BYTES` (float32 bytes of the centre tables), and a pipeline whose *two*
families both qualify is the only one that uses it at all — a partially dense ``measure`` would
have to broadcast the centre payload for the other family, and the row-wise merge is then the
honest route.  ``S = 150`` (``max_nodes = 50``, ``max_conns = 100``) gives ``22 501`` connection
cells, i.e. ~0.3 MB; ``S = 2000`` would need ~72 MB and falls back to the merge.

Determinism: the attribute term is reduced **in the centre's own row order**, exactly like the
merge's ``reduce_sum`` over the centre's gene axis (the pair distance is scattered into
``(P, M + 1)`` at the centre row each population gene homologues to, the dump column is sliced
off, and the surviving ``(P, M)`` block is reduced over axis 1).  Same values in the same
positions, so a dense ``measure`` is *bit-identical* to ``execute_distance`` on the same centre
and population — the property the equivalence gate asserts, not merely closeness.
"""

from typing import TYPE_CHECKING

import evoxc
import evoxc.numpy as enp

from ....common import I_INF
from .consts import ConstantPool
from .digest import _FOLDED_KEYS, _column, dense_slot, dense_width

if TYPE_CHECKING:  # pragma: no cover - typing only
    from ...base import BaseGenome
    from .base import BaseDistance

#: Upper bound on the float32 bytes of one family's centre tables (present/owner, attributes,
#: trailing-marker columns).  Above it the row-wise merge is the route: the merge's cost grows
#: with the gene axis ``M`` while a dense table grows with ``key_span``, so a large key span
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
    """
    width = dense_width(gene, int(genome.key_span))
    if width > (1 << 31) - 1:  # the dense index is an int32 gather/scatter index
        return False
    marker_columns = max(0, int(gene.key_width) - _FOLDED_KEYS)
    # owner (int32) + one float32 column per attribute + one int32 column per trailing marker
    bytes_per_cell = 4 * (1 + int(gene.attr_width) + marker_columns)
    return width * bytes_per_cell <= _DENSE_MAX_BYTES


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
    """
    node_keys, node_attrs, conn_keys, conn_attrs = payload
    if not (enabled(genome.node_gene, genome) and enabled(genome.conn_gene, genome)):
        return None
    node_distance = _family_distance(
        pool,
        state,
        genome,
        distance,
        genome.node_gene,
        node_keys,
        node_attrs,
        digest.node_dense,
        digest.node_real,
        digest.node_attrs,
        digest.node_count,
        digest.node_key_cols,
    )
    conn_distance = _family_distance(
        pool,
        state,
        genome,
        distance,
        genome.conn_gene,
        conn_keys,
        conn_attrs,
        digest.conn_dense,
        digest.conn_real,
        digest.conn_attrs,
        digest.conn_count,
        digest.conn_key_cols,
    )
    return evoxc.add(node_distance, conn_distance)


def _family_distance(
    pool: ConstantPool,
    state,
    genome: "BaseGenome",
    distance: "BaseDistance",
    gene,
    keys: evoxc.SymbolicTensor,
    attrs: evoxc.SymbolicTensor,
    pop_dense: evoxc.SymbolicTensor,
    pop_real: evoxc.SymbolicTensor,
    pop_attrs: evoxc.SymbolicTensor,
    pop_count: evoxc.SymbolicTensor,
    pop_key_cols: evoxc.SymbolicTensor,
) -> evoxc.SymbolicTensor:
    """One gene family of :func:`measure_centre`, against the digest's dense block.

    Args:
        pool (ConstantPool): Per-trace rank-0 constant pool.
        state: Host configuration container, forwarded to ``gene.distance``.
        genome ("BaseGenome"): Genome owning ``key_span`` (table geometry).
        distance ("BaseDistance"): Operator supplying the shared scalar tail.
        gene: Node or connection gene.
        keys (evoxc.SymbolicTensor): ``(1, M)`` or ``(1, M, K) int32`` centre key rows.
        attrs (evoxc.SymbolicTensor): ``(1, M, A) float32`` centre attribute rows.
        pop_dense (evoxc.SymbolicTensor): ``(P, M) int32`` digest slot per population gene, in
            the digest's own order.
        pop_real (evoxc.SymbolicTensor): ``(P, M) bool`` in-range population gene mask.
        pop_attrs (evoxc.SymbolicTensor): ``(P, M, A) float32`` population attributes in the
            digest's order.
        pop_count (evoxc.SymbolicTensor): ``(P,)`` population gene count of this family.
        pop_key_cols (evoxc.SymbolicTensor): ``(P, M, K) int32`` population key columns in the
            digest's order (read only for a gene with trailing marker columns).

    Returns:
        evoxc.SymbolicTensor: ``(P,) float32`` distances of this gene family.
    """
    width = dense_width(gene, int(genome.key_span))
    count = int(keys.shape[1])
    pop_size = int(pop_dense.shape[0])
    slots, _ = dense_slot(gene, int(genome.key_span), keys, pool)
    row = evoxc.reshape(slots, (count,))
    present = _present(keys, count, pool)
    # The centre table: one owner row index per slot.  A row that is absent (or whose key is
    # outside ``[0, S)``) writes ``count`` — the dump row of the scatter below — so a slot
    # without a usable centre gene can never be mistaken for one.
    owner = evoxc.scatter(
        enp.full((width,), count, dtype=evoxc.int32),
        row,
        evoxc.select(present, enp.arange(0, count, dtype=evoxc.int32), pool.i32(count)),
        axis=0,
    )
    owner_at = evoxc.gather(owner, pop_dense, axis=0)
    matched = evoxc.logical_and(evoxc.less(owner_at, pool.i32(count)), pop_real)
    marker_columns = int(gene.key_width) - _FOLDED_KEYS
    if marker_columns > 0:
        markers = evoxc.scatter(
            enp.zeros((width, marker_columns), dtype=evoxc.int32),
            row,
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
    attribute_distance = None
    attr_width = int(attrs.shape[-1])
    if attr_width:
        table = evoxc.scatter(
            enp.zeros((width, attr_width), dtype=evoxc.float32),
            row,
            evoxc.reshape(attrs, (count, attr_width)),
            axis=0,
        )
        centre_attrs = evoxc.gather(table, pop_dense, axis=0)
        pair = gene.distance(state, centre_attrs, pop_attrs)
        # ``pair`` holds the distance of each *population* gene against the centre row it
        # homologues to.  The scalar tail sums over the centre's rows, so the values are
        # written back into the centre's order (dump column for everything else) before the
        # reduction — that is what keeps the float summation order, and with it the result,
        # identical to the row-wise merge.
        attribute_distance = evoxc.reduce_sum(
            evoxc.slice(
                _reorder(pair, evoxc.select(matched, owner_at, pool.i32(count)), pop_size, count, pool),
                (0, 0),
                (pop_size, count),
            ),
            axes=1,
        )
    matched_count = evoxc.reduce_sum(evoxc.cast(matched, evoxc.int32), axes=1)
    centre_count = evoxc.reduce_sum(evoxc.cast(present, evoxc.int32), axes=0)
    return distance._scalar_distance(
        centre_count, matched_count, pop_count, attribute_distance, pool
    )


def _present(
    keys: evoxc.SymbolicTensor, count: int, pool: ConstantPool
) -> evoxc.SymbolicTensor:
    """Real-gene mask of one centre family, in payload row order.

    Args:
        keys: ``(1, M)`` or ``(1, M, K) int32`` centre key rows, ``I_INF`` padded.
        count: Static gene count ``M``.
        pool (ConstantPool): Per-trace rank-0 constant pool.

    Returns:
        ``(M,) bool`` True on rows that hold a gene — the same predicate the digest's
        ``count`` field uses, so the two sides of the formula count the same thing.

    Notes:
        Presence, not range: a gene whose key is outside ``[0, key_span)`` still *exists* in
        the payload and therefore still counts in ``cnt1`` (it can only fail to *match*,
        because the dense table has no slot for it).
    """
    column = keys if len(keys.shape) == 2 else _column(keys, 0)
    return evoxc.not_equal(evoxc.reshape(column, (count,)), pool.i32(I_INF))


def _reorder(
    values: evoxc.SymbolicTensor,
    column: evoxc.SymbolicTensor,
    pop_size: int,
    count: int,
    pool: ConstantPool,
) -> evoxc.SymbolicTensor:
    """Write each population gene's value into its centre row, one flat scatter.

    Args:
        values (evoxc.SymbolicTensor): ``(P, M) float32`` per-population-gene values.
        column (evoxc.SymbolicTensor): ``(P, M) int32`` centre row of each population gene,
            ``count`` where it has no usable homologue.
        pop_size (int): Static population batch ``P``.
        count (int): Static gene count ``M``.
        pool (ConstantPool): Per-trace rank-0 constant pool.

    Returns:
        ``(P, M + 1) float32`` — column ``j`` holds the value of the population gene that
        homologues with centre row ``j``, the trailing column is the dump slot.

    Notes:
        A flat scatter, not :func:`~tensorneat.common.tools.masked_flat_scatter`: every index
        is already inside ``[0, M]`` (the redirect to ``M`` happened in the caller's
        ``select``), so the generic out-of-range escape checks would be pure overhead.
        Duplicate targets are impossible — one centre row has one slot, so at most one
        population gene per genome can name it — and the dump column's content is never read.
    """
    stride = count + 1
    offsets = evoxc.reshape(
        enp.arange(0, pop_size * stride, stride, dtype=evoxc.int32), (pop_size, 1)
    )
    flat_index = evoxc.reshape(evoxc.add(offsets, column), (-1,))
    scattered = evoxc.scatter(
        enp.zeros((pop_size * stride,), dtype=evoxc.float32),
        flat_index,
        evoxc.reshape(values, (-1,)),
        axis=0,
    )
    return evoxc.reshape(scattered, (pop_size, stride))


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
