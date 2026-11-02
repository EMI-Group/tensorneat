"""Compatibility distance with disjoint/excess and attribute terms.

``distance = (non_homologous_count * compatibility_disjoint
+ homologous_attribute_distance * compatibility_weight) / max(cnt1, cnt2)``

Alignment of homologous genes is one sorted comparison per gene family (``O(M log M)``),
all counts are ``reduce_*`` over the gene axis, and the whole population of parent pairs is
resolved in a single batched kernel.

Both routes into this node — the pair-wise :meth:`DefaultDistance.gene_distance` and the
digest-driven :meth:`BaseDistance.gene_measure` — end in
:meth:`DefaultDistance.distance_from_alignment`, which is what makes them bit-identical
rather than merely close: the aligned attribute gather, the attribute term and the scalar
formula are literally the same code, run over the same rows in the same order.

Every rank-0 constant here comes from the caller's
:class:`~.consts.ConstantPool`, so the same weight or index costs one traced op per entry
point instead of one per use site.
"""

import math
from typing import TYPE_CHECKING

import evoxc
import evoxc.numpy as enp

from ....common import I_INF, ConfigError, State
from ...utils import valid_count
from ..alignment import find_homologous, key_rows
from .base import BaseDistance
from .consts import ConstantPool

if TYPE_CHECKING:  # pragma: no cover - typing only
    from ...base import BaseGenome
    from ...gene import BaseGene


def _pool(consts) -> ConstantPool:
    """Return the caller's constant pool, or a private one for a standalone call.

    Args:
        consts (ConstantPool | None): Pool threaded in from a top-level kernel entry.

    Returns:
        ConstantPool: ``consts`` unchanged, or a fresh empty pool.
    """
    return consts if consts is not None else ConstantPool()


def _weight(value: float, name: str) -> float:
    """Validate one distance weight on the host.

    Args:
        value: Caller-supplied weight; a negative or non-finite weight would make "more
            different" mean "closer" or poison every distance with ``NaN``.
        name: Constructor argument name, used in the error message.

    Returns:
        The weight as a Python ``float``.

    Raises:
        ConfigError: If ``value`` is negative or not finite.
    """
    weight = float(value)
    if not math.isfinite(weight) or weight < 0.0:
        raise ConfigError(
            f"DefaultDistance: {name} must be a finite non-negative weight, got {value!r}"
        )
    return weight


class DefaultDistance(BaseDistance):
    """Sum of node-gene and connection-gene compatibility distance.

    The two terms are additive: each gene family aligns its homologous keys, counts the genes
    without a homologue (disjoint + excess) and sums the per-gene attribute distance over the
    homologous pairs, then normalizes by the larger gene count so that pairs of structurally
    different sizes stay comparable.

    Args:
        compatibility_disjoint (float): Weight of the non-homologous gene count.
        compatibility_weight (float): Weight of the homologous attribute distance.

    Raises:
        TensorNEATError: If either weight is negative.
    """

    def __init__(
        self,
        compatibility_disjoint: float = 1.0,
        compatibility_weight: float = 0.4,
    ):
        """Store the two distance weights.

        Raises:
            ConfigError: If either weight is negative or not finite (a ``ConfigError`` is a
                ``TensorNEATError``).
        """
        self.compatibility_disjoint = _weight(
            compatibility_disjoint, "compatibility_disjoint"
        )
        self.compatibility_weight = _weight(compatibility_weight, "compatibility_weight")

    def __call__(
        self,
        state: State,
        genome: "BaseGenome",
        node_keys1: evoxc.SymbolicTensor,
        node_attrs1: evoxc.SymbolicTensor,
        conn_keys1: evoxc.SymbolicTensor,
        conn_attrs1: evoxc.SymbolicTensor,
        node_keys2: evoxc.SymbolicTensor,
        node_attrs2: evoxc.SymbolicTensor,
        conn_keys2: evoxc.SymbolicTensor,
        conn_attrs2: evoxc.SymbolicTensor,
    ) -> evoxc.SymbolicTensor:
        """Return ``node_distance + conn_distance`` for each parent pair.

        Args:
            state (State): Immutable host configuration container.
            genome ("BaseGenome"): Genome owning ``node_gene``/``conn_gene``.
            node_keys1 (evoxc.SymbolicTensor): ``(P, N, NK) int32`` node keys, parent 1,
                ``I_INF`` padded (``(P, N)`` for a one-column node key).
            node_attrs1 (evoxc.SymbolicTensor): ``(P, N, NA) float32`` node attrs, parent 1,
                NaN padded.
            conn_keys1 (evoxc.SymbolicTensor): ``(P, E, CK) int32`` conn keys, parent 1,
                ``I_INF`` padded.
            conn_attrs1 (evoxc.SymbolicTensor): ``(P, E, CA) float32`` conn attrs, parent 1,
                NaN padded.
            node_keys2 (evoxc.SymbolicTensor): ``(P, N, NK) int32`` node keys, parent 2,
                ``I_INF`` padded.
            node_attrs2 (evoxc.SymbolicTensor): ``(P, N, NA) float32`` node attrs, parent 2,
                NaN padded.
            conn_keys2 (evoxc.SymbolicTensor): ``(P, E, CK) int32`` conn keys, parent 2,
                ``I_INF`` padded.
            conn_attrs2 (evoxc.SymbolicTensor): ``(P, E, CA) float32`` conn attrs, parent 2,
                NaN padded.

        Returns:
            evoxc.SymbolicTensor: ``(P,) float32`` distances, one per parent pair.

        Raises:
            ConfigError: If the two parents' payload shapes differ (delegated to
                :meth:`BaseDistance.__call__`).
        """
        return super().__call__(
            state,
            genome,
            node_keys1,
            node_attrs1,
            conn_keys1,
            conn_attrs1,
            node_keys2,
            node_attrs2,
            conn_keys2,
            conn_attrs2,
        )

    def gene_distance(
        self,
        state: State,
        gene: "BaseGene",
        keys1: evoxc.SymbolicTensor,
        attrs1: evoxc.SymbolicTensor,
        keys2: evoxc.SymbolicTensor,
        attrs2: evoxc.SymbolicTensor,
        *,
        consts=None,
    ) -> evoxc.SymbolicTensor:
        """Compatibility distance of one gene family between two batches of genomes.

        Aligns homologous genes with a single sorted comparison of the concatenated key
        columns (``O(M log M)``, the batched equivalent of ``np.intersect1d`` in higher
        dimension), so a key present in both parents becomes one adjacent pair and every
        unmatched gene is counted as non-homologous.

        Args:
            state (State): Immutable host configuration container.
            gene ("BaseGene"): Node or connection gene supplying key width and
                ``distance``.
            keys1 (evoxc.SymbolicTensor): ``(P, M, K) int32`` key columns, parent 1, ``I_INF``
                padded; a node key may be spelled ``(P, M)`` (one column).
            attrs1 (evoxc.SymbolicTensor): ``(P, M, A) float32`` attributes, parent 1, NaN
                padded.
            keys2 (evoxc.SymbolicTensor): ``(P, M, K) int32`` key columns, parent 2, ``I_INF``
                padded.
            attrs2 (evoxc.SymbolicTensor): ``(P, M, A) float32`` attributes, parent 2, NaN
                padded.
            consts (ConstantPool | None): Pool for the rank-0 constants of this trace;
                ``None`` starts a private one.

        Returns:
            evoxc.SymbolicTensor: ``(P,) float32`` where each entry is
            ``(non_homologous_count * compatibility_disjoint
            + homologous_attribute_distance * compatibility_weight) / max(cnt1, cnt2)``,
            guarded to ``0`` for pairs where both genomes are empty in this gene family.

        Raises:
            ConfigError: If a key array is not 2-D/3-D.
            evoxc.ShapeError: Propagated unchanged if the two parents' key shapes differ —
                :meth:`__call__` rejects that on the host, so it can only surface on a direct
                call.

        Notes:
            The homologue table is :func:`~..alignment.find_homologous`; the rest of the
            arithmetic — the aligned attribute gather, the attribute term and the scalar
            formula — is :meth:`distance_from_alignment`, shared with the digest-driven
            :meth:`~..distance.base.BaseDistance.gene_measure`, so the pair-wise and the
            two-phase route are bit-identical by construction.
        """
        pool = _pool(consts)
        rows = find_homologous(key_rows(keys1), key_rows(keys2))
        return self.distance_from_rows(
            state,
            gene,
            attrs1,
            attrs2,
            rows,
            valid_count(keys1),
            valid_count(keys2),
            consts=pool,
        )

    def distance_from_rows(
        self,
        state: State,
        gene: "BaseGene",
        attrs1: evoxc.SymbolicTensor,
        attrs2: evoxc.SymbolicTensor,
        rows: evoxc.SymbolicTensor,
        cnt1: evoxc.SymbolicTensor,
        cnt2: evoxc.SymbolicTensor,
        *,
        consts=None,
    ) -> evoxc.SymbolicTensor:
        """Compatibility distance of one gene family from an ``I_INF``-padded row table.

        Args:
            state (State): Immutable host configuration container.
            gene ("BaseGene"): Node or connection gene supplying ``distance``.
            attrs1 (evoxc.SymbolicTensor): ``(P, M, A) float32`` attributes of parent 1, in
                *its own* row order.
            attrs2 (evoxc.SymbolicTensor): ``(P, M, A) float32`` attributes of parent 2, in
                *row-number order*: ``rows`` indexes this block directly.
            rows (evoxc.SymbolicTensor): ``(P, M) int32`` row of ``attrs2`` homologous to each
                gene of parent 1, ``I_INF`` where there is none.
            cnt1 (evoxc.SymbolicTensor): ``(P,)`` real genes of parent 1.
            cnt2 (evoxc.SymbolicTensor): ``(P,)`` real genes of parent 2.
            consts (ConstantPool | None): Pool for the rank-0 constants of this trace;
                ``None`` starts a private one.

        Returns:
            evoxc.SymbolicTensor: ``(P,) float32`` distances of this gene family.

        Notes:
            The documented row-table spelling: it derives the homology mask and the homologue
            count from ``rows`` and hands everything to :meth:`distance_from_alignment`.  The
            two-phase route already holds both, so it skips this step.
        """
        pool = _pool(consts)
        homologous = evoxc.not_equal(rows, pool.i32(I_INF))
        matched = evoxc.reduce_sum(homologous, axes=1)
        safe_rows = evoxc.select(homologous, rows, pool.i32(0))
        return self.distance_from_alignment(
            state,
            gene,
            attrs1,
            attrs2,
            safe_rows,
            homologous,
            cnt1,
            matched,
            cnt2,
            consts=pool,
        )

    def distance_from_alignment(
        self,
        state: State,
        gene: "BaseGene",
        attrs1: evoxc.SymbolicTensor,
        attrs2: evoxc.SymbolicTensor,
        rows: evoxc.SymbolicTensor,
        homologous: evoxc.SymbolicTensor,
        cnt1: evoxc.SymbolicTensor,
        matched: evoxc.SymbolicTensor,
        cnt2: evoxc.SymbolicTensor,
        *,
        consts=None,
    ) -> evoxc.SymbolicTensor:
        """Compatibility distance of one gene family from a finished alignment.

        Args:
            state (State): Immutable host configuration container.
            gene ("BaseGene"): Node or connection gene supplying ``distance``.
            attrs1 (evoxc.SymbolicTensor): ``(P, M, A) float32`` attributes of parent 1, in
                *its own* row order.
            attrs2 (evoxc.SymbolicTensor): ``(P, M, A) float32`` attributes of parent 2, in
                *row-number order*: ``rows`` indexes this block directly.
            rows (evoxc.SymbolicTensor): ``(P, M) int32`` row of ``attrs2`` homologous to each
                gene of parent 1, **redirected to row ``0``** where there is none.
            homologous (evoxc.SymbolicTensor): ``(P, M) bool`` True on the real homologues.
            cnt1 (evoxc.SymbolicTensor): ``(P,)`` real genes of parent 1.
            matched (evoxc.SymbolicTensor): ``(P,)`` number of homologous genes ``h``.
            cnt2 (evoxc.SymbolicTensor): ``(P,)`` real genes of parent 2.
            consts (ConstantPool | None): Pool for the rank-0 constants of this trace;
                ``None`` starts a private one.

        Returns:
            evoxc.SymbolicTensor: ``(P,) float32`` where each entry is
            ``(non_homologous_count * compatibility_disjoint
            + homologous_attribute_distance * compatibility_weight) / max(cnt1, cnt2)``,
            ``0`` for a pair where both genomes are empty in this gene family.

        Notes:
            The count ``non_homologous = cnt1 + cnt2 - 2 * h`` charges a key missing from
            either parent once, so *disjoint* and *excess* genes share the term.  The aligned
            attribute rows are fetched with **one unmasked** ``(P, M, A)`` gather: ``rows`` is
            already redirected to row ``0`` where it is absent, so no ``I_INF`` index reaches
            the backend and the per-gene distances are zeroed on the non-homologous slots
            **before** the ``reduce_sum`` — a ``NaN * 0`` product would otherwise poison the
            whole sum.  That masking is also what makes a ``NaN`` fill unnecessary: the
            gathered values at absent slots are never read.  The reduction runs over parent
            1's own row order in both routes, which is what keeps the float sum bit-identical.
            The empty-pair guard rides on the denominator: both terms of the numerator are
            exactly ``0.0`` there, while a non-empty pair's ``max(cnt1, cnt2)`` is at least
            ``1`` (counts are integers), so raising the denominator to ``1.0`` yields
            ``0.0 / 1.0 == 0.0`` instead of ``NaN`` — no second ``select`` over the result.
            A gene family with no attribute column (``attr_width == 0``, a zero-weight
            connection gene) has no attribute term at all: the gather is skipped and the
            distance is the structural ``disjoint/excess`` count alone.  Skipping is required,
            not an optimisation — a zero-width row stride is a ``0``-step ``arange``, which
            raises on the reference backend.
        """
        pool = _pool(consts)
        pop_size, count = int(attrs1.shape[0]), int(attrs1.shape[1])
        attr_width = int(attrs1.shape[-1])

        non_homologous = evoxc.subtract(
            evoxc.add(cnt1, cnt2),
            evoxc.multiply(pool.i64(2), matched),
        )

        attribute_distance = None
        if attr_width:
            # Attribute row of parent 2 behind every gene of parent 1: the row offset is
            # ``row * A`` and the ``(P, M, A)`` block adds the within-row column.  ``rows`` is
            # already redirected to row ``0`` where a gene has no homologue, so no ``I_INF``
            # index reaches the backend and the flat index needs no range check of its own.
            row_offsets = enp.reshape(
                evoxc.multiply(rows, pool.i32(attr_width)), (pop_size, count, 1)
            )
            columns = enp.arange(0, attr_width, dtype=evoxc.int32)
            # Row stride as a ``(P, 1, 1)`` block: ``batch_offsets`` returns ``(P, 1)``, which
            # would broadcast against the last two axes of the ``(P, M, A)`` index block
            # instead of the leading one.
            rows_stride = enp.reshape(
                enp.arange(
                    0, pop_size * count * attr_width, count * attr_width, dtype=evoxc.int32
                ),
                (pop_size, 1, 1),
            )
            index = evoxc.add(evoxc.add(row_offsets, columns), rows_stride)
            homologous_attrs = evoxc.reshape(
                evoxc.gather(
                    evoxc.reshape(attrs2, (pop_size * count * attr_width,)), index, axis=0
                ),
                (pop_size, count, attr_width),
            )
            pair_distance = gene.distance(state, attrs1, homologous_attrs)
            attribute_distance = evoxc.reduce_sum(
                evoxc.select(homologous, pair_distance, pool.f32(0.0)), axes=1
            )

        # Counts are small integers, so ``float32(max(cnt1, cnt2))`` is exact and the
        # maximum is cheaper on the integer dtypes than on two casts.
        denominator = evoxc.cast(evoxc.maximum(cnt1, cnt2), evoxc.float32)
        numerator = evoxc.multiply(
            evoxc.cast(non_homologous, evoxc.float32),
            pool.f32(self.compatibility_disjoint),
        )
        if attribute_distance is not None:
            numerator = evoxc.add(
                numerator,
                evoxc.multiply(
                    attribute_distance, pool.f32(self.compatibility_weight)
                ),
            )
        return evoxc.divide(
            numerator, evoxc.maximum(denominator, pool.f32(1.0))
        )


__all__ = ["DefaultDistance"]
