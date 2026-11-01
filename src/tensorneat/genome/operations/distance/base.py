"""Compatibility distance contract: one scalar distance per parent pair, batch-first.

The algorithm forms the pairs, so the kernel sees a population of parent pairs on axis 0 and
returns ``(P,)`` distances.  Alignment of homologous genes is done with one sort per gene
family (never an ``O(M^2)`` scan) and every count is a ``reduce_*`` over the gene axis; no
operator loops over individuals and nothing here is differentiable.
"""

from typing import TYPE_CHECKING

import evoxc

from ....common import ConfigError, State, StatefulBaseClass

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
