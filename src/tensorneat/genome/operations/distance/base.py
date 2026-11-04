"""Compatibility distance contract: one scalar distance per parent pair, batch-first.

The algorithm forms the pairs, so the kernel sees a population of parent pairs on axis 0 and
returns ``(P,)`` distances.  Alignment of homologous genes is done with one sort per gene
family (never an ``O(M^2)`` scan) and every count is a ``reduce_*`` over the gene axis; no
operator loops over individuals and nothing here is differentiable.

Speciation measures one *centre* genome against the whole population twice per generation, so
the same population payload would be re-sorted ``2 * species_size`` times.
The two-phase API removes that: :meth:`BaseDistance.prepare` sorts each gene family of the
population once per generation into a :class:`~.digest.DistanceDigest`, and
:meth:`BaseDistance.measure` merges one centre genome against that digest per call, producing
the very same numbers as :meth:`BaseDistance.__call__` (bit-identical: both routes share
:meth:`~.default.DefaultDistance.distance_from_rows`, so the aligned attribute gather, the
attribute term and the scalar formula are literally the same code).
"""

from typing import TYPE_CHECKING

import evoxc
import evoxc.numpy as enp

from ....common import ConfigError, I_INF, State, StatefulBaseClass
from . import dense as _dense
from . import digest as _digest
from .consts import ConstantPool
from .digest import DistanceDigest

if TYPE_CHECKING:  # pragma: no cover - typing only; keeps genome -> operations acyclic
    from ...base import BaseGenome

#: Field names of one parent's four-array payload, in ``GenomeTensors`` order.
_PAYLOAD_FIELDS = ("node_keys", "node_attrs", "conn_keys", "conn_attrs")

#: Static ranks of a payload array.  A node key row is one column wide and is spelled
#: ``(P, M)`` by ``GenomeTensors`` and ``(P, M, 1)`` by this kernel's key-column
#: contract, so both ranks are legal for a key array; an attribute row is always
#: ``(P, M, A)``.
_KEY_RANKS = (2, 3)
_ATTR_RANK = 3


class BaseDistance(StatefulBaseClass):
    """Pairwise compatibility distance between two batches of genomes.

    The result is additive over gene families — node distance plus connection distance — and
    normalized per pair, so pairs with different gene counts stay comparable and speciation
    thresholds are meaningful across a population.  Empty pairs contribute ``0`` rather than
    ``NaN``.

    Subclasses own the single-family hook ``gene_distance`` (the disjoint/excess count, the
    alignment of homologous genes and the normalization); this class only adds the two
    families it feeds with the genome's genes together, so no driver is duplicated.
    The same split holds for the two-phase API: :meth:`prepare` and :meth:`measure` add the
    two families exactly like :meth:`__call__`, and the per-family tail
    (:meth:`gene_measure`) ends in the subclass hook ``distance_from_rows`` — the second half
    of ``gene_distance`` — so a subclass states the arithmetic once.
    """

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
        """Compute one compatibility distance per parent pair.

        Args:
            state (State): Immutable host configuration container.
            genome ("BaseGenome"): Genome owning ``node_gene``/``conn_gene`` (their key
                widths and ``distance`` hooks).
            node_keys1 (evoxc.SymbolicTensor): ``(P, N, NK) int32`` node keys of parent 1,
                ``I_INF`` padded (``(P, N)`` for a one-column node key).
            node_attrs1 (evoxc.SymbolicTensor): ``(P, N, NA) float32`` node attrs of parent
                1, NaN padded.
            conn_keys1 (evoxc.SymbolicTensor): ``(P, E, CK) int32`` conn keys of parent 1,
                ``I_INF`` padded.
            conn_attrs1 (evoxc.SymbolicTensor): ``(P, E, CA) float32`` conn attrs of parent
                1, NaN padded.
            node_keys2 (evoxc.SymbolicTensor): ``(P, N, NK) int32`` node keys of parent 2,
                ``I_INF`` padded.
            node_attrs2 (evoxc.SymbolicTensor): ``(P, N, NA) float32`` node attrs of parent
                2, NaN padded.
            conn_keys2 (evoxc.SymbolicTensor): ``(P, E, CK) int32`` conn keys of parent 2,
                ``I_INF`` padded.
            conn_attrs2 (evoxc.SymbolicTensor): ``(P, E, CA) float32`` conn attrs of parent
                2, NaN padded.

        Returns:
            evoxc.SymbolicTensor: ``(P,) float32`` compatibility distances for the batch of
            parent pairs.

        Raises:
            ConfigError: If the two parents' payload shapes differ, or if one parent's key
                and attribute arrays disagree on ``(P, M)``.
        """
        payload1 = (node_keys1, node_attrs1, conn_keys1, conn_attrs1)
        payload2 = (node_keys2, node_attrs2, conn_keys2, conn_attrs2)
        _validate_payload(type(self).__name__, payload1, payload2)

        node_distance = self.gene_distance(
            state, genome.node_gene, node_keys1, node_attrs1, node_keys2, node_attrs2
        )
        conn_distance = self.gene_distance(
            state, genome.conn_gene, conn_keys1, conn_attrs1, conn_keys2, conn_attrs2
        )
        return evoxc.add(node_distance, conn_distance)

    def prepare(
        self,
        state: State,
        genome: "BaseGenome",
        node_keys: evoxc.SymbolicTensor,
        node_attrs: evoxc.SymbolicTensor,
        conn_keys: evoxc.SymbolicTensor,
        conn_attrs: evoxc.SymbolicTensor,
        *,
        consts=None,
    ) -> DistanceDigest:
        """Sort one population payload into the reusable distance digest.

        Args:
            state (State): Immutable host configuration container.
            genome ("BaseGenome"): Genome owning ``node_gene``/``conn_gene`` (their key
                widths and ``distance`` hooks).
            node_keys (evoxc.SymbolicTensor): ``(P, N) int32`` node keys, ``I_INF`` padded.
            node_attrs (evoxc.SymbolicTensor): ``(P, N, NA) float32`` node attrs, NaN padded.
            conn_keys (evoxc.SymbolicTensor): ``(P, E, CK) int32`` conn keys, ``I_INF``
                padded.
            conn_attrs (evoxc.SymbolicTensor): ``(P, E, CA) float32`` conn attrs, NaN padded.
            consts (ConstantPool | None): Pool for the rank-0 constants of this trace; pass
                one pool to every kernel of a generation so each constant costs a single op.
                ``None`` starts a private one.

        Returns:
            DistanceDigest: one sorted, tagged gene family per payload family — valid once per
            generation, for any number of :meth:`measure` calls — plus the two out-of-range
            gene counts ``node_oor``/``conn_oor``: present genes whose key lies outside
            ``[0, genome.key_span)``.  A non-zero count is the documented precondition failure
            of the single-centre fast path (the dense table cannot address those genes and
            treats them as absent, while the merge matches them exactly); the pipeline cannot
            raise on it, so a caller that relies on the fast path checks the leaves on the host,
            or configures ``DefaultDistance(dense_single_centre=False)``.

        Raises:
            ConfigError: If a key array is not 2-D/3-D, an attribute array is not 3-D, a key
                array and its attribute array disagree on ``(P, M)``, or the key width is not
                the gene's.

        Notes:
            A plain traced helper, never an ``evoxc.defn``: it is called from inside the
            caller's graph (a defn cannot call another defn).
        """
        payload = (node_keys, node_attrs, conn_keys, conn_attrs)
        _validate_single(type(self).__name__, "prepare", payload)
        pool = consts if consts is not None else ConstantPool()
        key_span = int(genome.key_span)
        # The single-centre fast path's statics are built here, once, instead of once per
        # ``measure`` call.  The gate is the same static predicate the route itself checks, so a
        # digest never carries statics its consumer would not use, and a merge-only
        # configuration (a large ``key_span``) never materialises the big table bases.
        dense = _dense.enabled(genome.node_gene, genome) and _dense.enabled(
            genome.conn_gene, genome
        )
        tail = self._tail_constants()
        node = _digest.prepare_family(
            state,
            genome.node_gene,
            node_keys,
            node_attrs,
            key_span=key_span,
            consts=pool,
            dense=dense,
            tail=tail,
        )
        conn = _digest.prepare_family(
            state,
            genome.conn_gene,
            conn_keys,
            conn_attrs,
            key_span=key_span,
            consts=pool,
            dense=dense,
            tail=tail,
        )
        # The fast path's ``cnt2`` pair, stacked once per generation instead of once per
        # ``measure`` call (the route traces its scalar tail once for both families).
        rows = int(node.count.shape[0])
        tail_counts = (
            evoxc.concatenate(
                (
                    evoxc.reshape(node.count, (1, rows)),
                    evoxc.reshape(conn.count, (1, rows)),
                ),
                axis=0,
            )
            if dense
            else enp.zeros((0, 0), dtype=evoxc.int32)
        )
        return DistanceDigest(
            node.keys,
            node.attrs,
            node.count,
            node.key_cols,
            conn.keys,
            conn.attrs,
            conn.count,
            conn.key_cols,
            node.dense,
            node.real,
            node.oor,
            conn.dense,
            conn.real,
            conn.oor,
            node.layout,
            conn.layout,
            tail_counts,
            node.statics,
            conn.statics,
        )

    def measure(
        self,
        state: State,
        genome: "BaseGenome",
        node_keys: evoxc.SymbolicTensor,
        node_attrs: evoxc.SymbolicTensor,
        conn_keys: evoxc.SymbolicTensor,
        conn_attrs: evoxc.SymbolicTensor,
        digest: DistanceDigest,
        *,
        consts=None,
    ) -> evoxc.SymbolicTensor:
        """Distance of one centre genome against every genome of a prepared digest.

        Args:
            state (State): Immutable host configuration container.
            genome ("BaseGenome"): Genome owning ``node_gene``/``conn_gene``.
            node_keys (evoxc.SymbolicTensor): ``(P, N) int32`` centre node keys — batch ``P``
                matching the digest, or batch ``1`` (one centre) which is broadcast.
            node_attrs (evoxc.SymbolicTensor): ``(P, N, NA) float32`` centre node attrs.
            conn_keys (evoxc.SymbolicTensor): ``(P, E, CK) int32`` centre conn keys.
            conn_attrs (evoxc.SymbolicTensor): ``(P, E, CA) float32`` centre conn attrs.
            digest (DistanceDigest): Population digest from :meth:`prepare`.
            consts (ConstantPool | None): Pool for the rank-0 constants of this trace; pass
                one pool to every kernel of a generation so each constant costs a single op.
                ``None`` starts a private one.

        Returns:
            evoxc.SymbolicTensor: ``(P,) float32`` compatibility distances, ``0`` for an empty
            pair — bit-identical to :meth:`__call__` on the same payloads.

        Raises:
            ConfigError: If the centre's static shapes are invalid, if its batch is neither
                ``1`` nor the digest's, or if the digest's gene axes do not match the centre's.

        Notes:
            A plain traced helper (see :meth:`prepare`).  A batch-1 centre is broadcast to the
            digest's batch up front, so the per-family kernels are batch-first throughout.
        """
        payload = (node_keys, node_attrs, conn_keys, conn_attrs)
        _validate_single(type(self).__name__, "measure", payload)
        batch = _validate_digest(type(self).__name__, payload, digest)
        pool = consts if consts is not None else ConstantPool()
        if int(node_keys.shape[0]) == 1:
            fast = self._measure_single_centre(pool, state, genome, payload, digest)
            if fast is not None:
                return fast
        if int(node_keys.shape[0]) != batch:
            payload = _broadcast_payload(payload, batch)
            node_keys, node_attrs, conn_keys, conn_attrs = payload
        node_distance = self._measure_family(
            pool,
            state,
            genome.node_gene,
            node_keys,
            node_attrs,
            digest.node_keys,
            digest.node_attrs,
            digest.node_count,
            digest.node_key_cols,
            digest.node_statics,
        )
        conn_distance = self._measure_family(
            pool,
            state,
            genome.conn_gene,
            conn_keys,
            conn_attrs,
            digest.conn_keys,
            digest.conn_attrs,
            digest.conn_count,
            digest.conn_key_cols,
            digest.conn_statics,
        )
        return evoxc.add(node_distance, conn_distance)

    def _measure_single_centre(
        self,
        pool: ConstantPool,
        state: State,
        genome: "BaseGenome",
        payload: tuple,
        digest: DistanceDigest,
    ) -> evoxc.SymbolicTensor | None:
        """Hook: distance of one batch-1 centre against the digest, or ``None``.

        Args:
            pool (ConstantPool): Per-trace rank-0 constant pool.
            state (State): Immutable host configuration container.
            genome ("BaseGenome"): Genome owning ``node_gene``/``conn_gene``.
            payload (tuple): The four centre arrays, batch ``1``.
            digest (DistanceDigest): Population digest from :meth:`prepare`.

        Returns:
            ``(P,) float32`` distances, or ``None`` when this operator has no single-centre
            route — the caller then broadcasts the centre and runs the row-wise merge.

        Notes:
            A static capability decision, taken before any op of the row-wise route is
            emitted, so a subclass either answers the whole call or none of it.  The base
            contract has no dense table, so it answers ``None``; the shipped
            :class:`~.default.DefaultDistance` answers with :mod:`.dense` while the gene
            geometry fits that module's static gate.
        """
        return None

    def gene_measure(
        self,
        state: State,
        gene,
        keys: evoxc.SymbolicTensor,
        attrs: evoxc.SymbolicTensor,
        digest_keys: evoxc.SymbolicTensor,
        digest_attrs: evoxc.SymbolicTensor,
        digest_count: evoxc.SymbolicTensor,
        digest_key_cols: evoxc.SymbolicTensor,
    ) -> evoxc.SymbolicTensor:
        """Compatibility distance of one gene family against a population digest.

        Args:
            state (State): Immutable host configuration container.
            gene: Node or connection gene supplying key width and ``distance``.
            keys (evoxc.SymbolicTensor): ``(P, M)`` or ``(P, M, K) int32`` centre key array.
            attrs (evoxc.SymbolicTensor): ``(P, M, A) float32`` centre attribute rows.
            digest_keys (evoxc.SymbolicTensor): ``(P, M) int64`` composites of this family.
            digest_attrs (evoxc.SymbolicTensor): ``(P, M, A) float32`` population attributes
                permuted into the digest's gene order.
            digest_count (evoxc.SymbolicTensor): ``(P,) int32`` real genes per individual.
            digest_key_cols (evoxc.SymbolicTensor): ``(P, M, K) int32`` population key columns
                in the digest's gene order — the exact-match oracle for a gene whose key
                width exceeds the number of columns the composite folds.

        Returns:
            evoxc.SymbolicTensor: ``(P,) float32`` distances of this gene family, disjoint /
            excess count and normalization included.

        Notes:
            The homologue table is the merge in :func:`~.digest.measure_alignment` (one sort
            per call, no population re-sort); everything after it is
            :meth:`distance_from_alignment`, the very tail ``gene_distance`` runs — which is
            what makes the two routes bit-identical rather than merely close.
            A standalone call, so it starts its own constant pool; :meth:`measure` shares one
            pool across both families and across every call of a generation.
        """
        return self._measure_family(
            ConstantPool(),
            state,
            gene,
            keys,
            attrs,
            digest_keys,
            digest_attrs,
            digest_count,
            digest_key_cols,
        )

    def _measure_family(
        self,
        pool: ConstantPool,
        state: State,
        gene,
        keys: evoxc.SymbolicTensor,
        attrs: evoxc.SymbolicTensor,
        digest_keys: evoxc.SymbolicTensor,
        digest_attrs: evoxc.SymbolicTensor,
        digest_count: evoxc.SymbolicTensor,
        digest_key_cols: evoxc.SymbolicTensor,
        statics,
    ) -> evoxc.SymbolicTensor:
        """One gene family of :meth:`measure`, with the pool and the alignment threaded in.

        Args:
            pool (ConstantPool): Pool for the rank-0 constants of this trace.
            state (State): Immutable host configuration container.
            gene: Node or connection gene supplying key width and ``distance``.
            keys (evoxc.SymbolicTensor): ``(P, M)`` or ``(P, M, K) int32`` centre key array.
            attrs (evoxc.SymbolicTensor): ``(P, M, A) float32`` centre attribute rows.
            digest_keys (evoxc.SymbolicTensor): ``(P, M) int64`` composites of this family.
            digest_attrs (evoxc.SymbolicTensor): ``(P, M, A) float32`` population attributes
                permuted into the digest's gene order.
            digest_count (evoxc.SymbolicTensor): ``(P,) int32`` real genes per individual.
            digest_key_cols (evoxc.SymbolicTensor): ``(P, M, K) int32`` population key columns
                in the digest's gene order.
            statics (MeasureStatics): The family's per-generation scaffolding from the digest —
                the rank-0 scalars and the two flat index blocks, so this call traces neither a
                ``constant`` nor an ``arange`` of its own.

        Returns:
            evoxc.SymbolicTensor: ``(P,) float32`` distances of this gene family.

        Notes:
            The merge hands back the homology mask and both counts, so nothing downstream
            re-derives a mask from an ``I_INF`` row table or re-counts the centre's genes.
        """
        aligned = _digest.measure_alignment(
            gene, keys, digest_keys, digest_key_cols, statics=statics, consts=pool
        )
        return self.distance_from_alignment(
            state,
            gene,
            attrs,
            digest_attrs,
            aligned.rows,
            aligned.homologous,
            aligned.count,
            aligned.matched,
            digest_count,
            statics=statics.attr,
            consts=pool,
        )

    def distance_from_alignment(
        self,
        state: State,
        gene,
        attrs1: evoxc.SymbolicTensor,
        attrs2: evoxc.SymbolicTensor,
        rows: evoxc.SymbolicTensor,
        homologous: evoxc.SymbolicTensor,
        cnt1: evoxc.SymbolicTensor,
        matched: evoxc.SymbolicTensor,
        cnt2: evoxc.SymbolicTensor,
        *,
        statics=None,
        consts=None,
    ) -> evoxc.SymbolicTensor:
        """Distance of one gene family from a finished alignment (subclass hook).

        Args:
            state (State): Immutable host configuration container.
            gene: Node or connection gene supplying ``distance``.
            attrs1 (evoxc.SymbolicTensor): ``(P, M, A) float32`` attributes of parent 1.
            attrs2 (evoxc.SymbolicTensor): ``(P, M, A) float32`` attributes of parent 2, in
                row-number order.
            rows (evoxc.SymbolicTensor): ``(P, M) int32`` homologue row per parent-1 gene,
                redirected to row ``0`` where there is none.
            homologous (evoxc.SymbolicTensor): ``(P, M) bool`` True on the real homologues.
            cnt1 (evoxc.SymbolicTensor): ``(P,)`` real genes of parent 1.
            matched (evoxc.SymbolicTensor): ``(P,)`` number of homologous genes.
            cnt2 (evoxc.SymbolicTensor): ``(P,)`` real genes of parent 2.
            statics (AttrStatics | None): The family's per-generation attribute scaffold when
                the caller has a digest (the row-wise ``measure`` route); the pair-wise route
                passes ``None`` and builds what it needs per call.
            consts (ConstantPool | None): Pool for the rank-0 constants of this trace.

        Returns:
            evoxc.SymbolicTensor: ``(P,) float32`` distances of this gene family.

        Notes:
            Default adapter for a subclass that implements only
            :meth:`distance_from_rows`: it re-spells the aligned rows as the documented
            ``I_INF``-padded table and calls that hook, which costs one extra ``select`` and
            one extra mask derivation.  :class:`~.default.DefaultDistance`, the only
            implementation in this package, overrides it with the shared tail and pays
            neither.
        """
        pool = consts if consts is not None else ConstantPool()
        rows_inf = evoxc.select(homologous, rows, pool.i32(I_INF))
        return self.distance_from_rows(
            state, gene, attrs1, attrs2, rows_inf, cnt1, cnt2
        )

    def _tail_constants(self) -> tuple[float, float, float] | None:
        """Constants the scalar tail multiplies by, or ``None`` for an operator without them.

        Returns:
            tuple[float, float, float] | None: ``(compatibility_disjoint, compatibility_weight,
            1.0)`` when the subclass has a fixed scalar tail, else ``None``.

        Notes:
            Hoisted through the digest so the ``2 * species_size`` measures of a generation
            share one rank-0 constant each instead of emitting three per call: ``evoxc`` never
            deduplicates constants, and on the reference interpreter a rank-0 constant costs one
            traced op like any other.  Both routes read them off
            :attr:`~.layout.MeasureStatics.tail`; only a standalone call with no digest builds
            them from its own pool.
        """
        return None


def _validate_single(name: str, caller: str, payload: tuple) -> None:
    """Reject a payload whose static shapes the two-phase API cannot consume.

    Args:
        name: Operator class name, used in the error message.
        caller: Method name, so the message names the entry point that refused.
        payload: The four payload arrays, in ``GenomeTensors`` order.

    Raises:
        ConfigError: If a key array is not 2-D/3-D, an attribute array is not 3-D, or a key
            array and its attribute array disagree on ``(P, M)``.

    Notes:
        Pure host validation of build-time properties: shapes are static, so catching a
        mismatch here names the offending field instead of letting a reshape read a broadcast
        pair.
    """
    for prefix, keys, attrs in (
        ("node", payload[0], payload[1]),
        ("conn", payload[2], payload[3]),
    ):
        if len(keys.shape) not in _KEY_RANKS:
            raise ConfigError(
                f"{name}.{caller}: {prefix} key array must be (P, M) or (P, M, K), "
                f"got shape {tuple(keys.shape)}"
            )
        if len(attrs.shape) != _ATTR_RANK:
            raise ConfigError(
                f"{name}.{caller}: {prefix} attribute array must be (P, M, A), "
                f"got shape {tuple(attrs.shape)}"
            )
        if tuple(keys.shape[:2]) != tuple(attrs.shape[:2]):
            raise ConfigError(
                f"{name}.{caller}: {prefix} keys {tuple(keys.shape)} and attributes "
                f"{tuple(attrs.shape)} disagree on (P, M)"
            )


def _validate_digest(name: str, payload: tuple, digest: DistanceDigest) -> int:
    """Check a centre payload against a prepared digest and return the batch width.

    Args:
        name: Operator class name, used in the error message.
        payload: The four centre payload arrays, in ``GenomeTensors`` order.
        digest (DistanceDigest): Population digest from :meth:`BaseDistance.prepare`.

    Returns:
        int: The digest's population batch ``P`` — the centre must reach it by broadcast.

    Raises:
        ConfigError: If ``digest`` is not a :class:`DistanceDigest`, if its gene or attribute
            axes do not match the centre's, or if the centre's batch is neither ``1`` nor ``P``.
        The population axis is deliberately *not* compared: a batch-1 centre is legal and is
        broadcast to the digest's batch by :meth:`BaseDistance.measure`.

    Notes:
        The digest is per generation and reused for every centre, so a mismatched digest is a
        configuration error at the call site, never a shape error deep inside a gather.
    """
    if not isinstance(digest, DistanceDigest):
        raise ConfigError(
            f"{name}.measure: expected a DistanceDigest from prepare(), "
            f"got {type(digest).__name__}"
        )
    batch = int(digest.node_count.shape[0])
    for (
        family,
        centre_keys,
        centre_attrs,
        digest_keys,
        digest_attrs,
        digest_count,
        digest_key_cols,
        digest_dense,
        digest_real,
    ) in (
        (
            "node",
            payload[0],
            payload[1],
            digest.node_keys,
            digest.node_attrs,
            digest.node_count,
            digest.node_key_cols,
            digest.node_dense,
            digest.node_real,
        ),
        (
            "conn",
            payload[2],
            payload[3],
            digest.conn_keys,
            digest.conn_attrs,
            digest.conn_count,
            digest.conn_key_cols,
            digest.conn_dense,
            digest.conn_real,
        ),
    ):
        if int(digest_keys.shape[1]) != int(centre_keys.shape[1]):
            raise ConfigError(
                f"{name}.measure: the {family} digest holds {int(digest_keys.shape[1])} "
                f"gene rows but the centre's key array is {tuple(centre_keys.shape)}"
            )
        if tuple(int(size) for size in digest_attrs.shape[1:]) != tuple(
            int(size) for size in centre_attrs.shape[1:]
        ):
            raise ConfigError(
                f"{name}.measure: the {family} digest attributes are "
                f"{(int(digest_attrs.shape[1]), int(digest_attrs.shape[2]))} (M, A) but the "
                f"centre's are {(int(centre_attrs.shape[1]), int(centre_attrs.shape[2]))}"
            )
        if len(digest_count.shape) != 1:
            raise ConfigError(
                f"{name}.measure: the {family} digest count must be (P,), got "
                f"{tuple(digest_count.shape)}"
            )
        if int(digest_key_cols.shape[1]) != int(centre_keys.shape[1]):
            raise ConfigError(
                f"{name}.measure: the {family} digest key columns hold "
                f"{int(digest_key_cols.shape[1])} gene rows but the centre's key array is "
                f"{tuple(centre_keys.shape)}"
            )
        for field, block in (("dense slot", digest_dense), ("real-gene mask", digest_real)):
            if tuple(int(size) for size in block.shape) != (
                batch,
                int(centre_keys.shape[1]),
            ):
                raise ConfigError(
                    f"{name}.measure: the {family} digest {field} is "
                    f"{tuple(block.shape)} but the digest holds {batch} genomes of "
                    f"{int(centre_keys.shape[1])} gene rows"
                )
    if int(digest.conn_count.shape[0]) != batch:
        raise ConfigError(
            f"{name}.measure: the two digest families disagree on the population batch "
            f"({batch} vs {int(digest.conn_count.shape[0])})"
        )
    centre_batch = int(payload[0].shape[0])
    if centre_batch not in (1, batch):
        raise ConfigError(
            f"{name}.measure: the centre batch is {centre_batch} but the digest holds "
            f"{batch} genomes; pass one centre (batch 1) or a digest-row per centre"
        )
    return batch


def _broadcast_payload(payload: tuple, batch: int) -> tuple:
    """Broadcast a batch-1 payload to ``batch`` rows, field by field.

    Args:
        payload: The four payload arrays of one centre genome, batch ``1``.
        batch: Target population batch ``P``.

    Returns:
        The four arrays with population axis ``batch``.  Every row is the same genome, so the
        values downstream are unchanged.
    """
    return tuple(
        enp.broadcast_to(array, (batch,) + tuple(array.shape[1:])) for array in payload
    )


def _validate_payload(name: str, payload1: tuple, payload2: tuple) -> None:
    """Reject two parents that cannot be compared, before any graph op is emitted.

    Args:
        name: Operator class name, used in the error message.
        payload1: The four payload arrays of parent 1, in ``GenomeTensors`` order.
        payload2: The four payload arrays of parent 2, in the same order.

    Raises:
        ConfigError: If a field's two shapes differ, if a key array is not 2-D/3-D, if an
            attribute array is not 3-D, or if a key array and its attribute array disagree
            on ``(P, M)``.

    Notes:
        Shapes are static at build time, so a mismatch is a configuration error rather than a
        backend error: catching it here names the offending field instead of letting the
        alignment sort read a broadcast pair.
    """
    for field, first, second in zip(_PAYLOAD_FIELDS, payload1, payload2):
        if tuple(first.shape) != tuple(second.shape):
            raise ConfigError(
                f"{name}: the two parents' {field} shapes differ "
                f"({tuple(first.shape)} vs {tuple(second.shape)}); both parents of a pair "
                "must share one static shape"
            )
    for prefix, keys, attrs in (
        ("node", payload1[0], payload1[1]),
        ("conn", payload1[2], payload1[3]),
        ("node", payload2[0], payload2[1]),
        ("conn", payload2[2], payload2[3]),
    ):
        if len(keys.shape) not in _KEY_RANKS:
            raise ConfigError(
                f"{name}: {prefix} key array must be (P, M) or (P, M, K), "
                f"got shape {tuple(keys.shape)}"
            )
        if len(attrs.shape) != _ATTR_RANK:
            raise ConfigError(
                f"{name}: {prefix} attribute array must be (P, M, A), "
                f"got shape {tuple(attrs.shape)}"
            )
        if tuple(keys.shape[:2]) != tuple(attrs.shape[:2]):
            raise ConfigError(
                f"{name}: {prefix} keys {tuple(keys.shape)} and attributes "
                f"{tuple(attrs.shape)} disagree on (P, M)"
            )


__all__ = ["BaseDistance"]
