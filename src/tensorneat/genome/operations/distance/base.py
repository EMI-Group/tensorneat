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

from ....common import ConfigError, State, StatefulBaseClass
from ...utils import valid_count
from . import digest as _digest
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

        Returns:
            DistanceDigest: one sorted, tagged gene family per payload family — valid once per
            generation, for any number of :meth:`measure` calls.

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
        node = _digest.prepare_family(state, genome.node_gene, node_keys, node_attrs)
        conn = _digest.prepare_family(state, genome.conn_gene, conn_keys, conn_attrs)
        return DistanceDigest(
            node.keys,
            node.attrs,
            node.count,
            node.key_cols,
            conn.keys,
            conn.attrs,
            conn.count,
            conn.key_cols,
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
        if int(node_keys.shape[0]) != batch:
            payload = _broadcast_payload(payload, batch)
            node_keys, node_attrs, conn_keys, conn_attrs = payload
        node_distance = self.gene_measure(
            state,
            genome.node_gene,
            node_keys,
            node_attrs,
            digest.node_keys,
            digest.node_attrs,
            digest.node_count,
            digest.node_key_cols,
        )
        conn_distance = self.gene_measure(
            state,
            genome.conn_gene,
            conn_keys,
            conn_attrs,
            digest.conn_keys,
            digest.conn_attrs,
            digest.conn_count,
            digest.conn_key_cols,
        )
        return evoxc.add(node_distance, conn_distance)

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
            The homologue table is the merge in :func:`~.digest.match_rows` (one sort per
            call, no population re-sort); everything after it is ``distance_from_rows``, the
            very tail ``gene_distance`` runs — which is what makes the two routes
            bit-identical rather than merely close.
        """
        rows = _digest.match_rows(gene, keys, digest_keys, digest_key_cols)
        return self.distance_from_rows(
            state, gene, attrs, digest_attrs, rows, valid_count(keys), digest_count
        )



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
    for (
        family,
        centre_keys,
        centre_attrs,
        digest_keys,
        digest_attrs,
        digest_count,
        digest_key_cols,
    ) in (
        (
            "node",
            payload[0],
            payload[1],
            digest.node_keys,
            digest.node_attrs,
            digest.node_count,
            digest.node_key_cols,
        ),
        (
            "conn",
            payload[2],
            payload[3],
            digest.conn_keys,
            digest.conn_attrs,
            digest.conn_count,
            digest.conn_key_cols,
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
    batch = int(digest.node_count.shape[0])
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
