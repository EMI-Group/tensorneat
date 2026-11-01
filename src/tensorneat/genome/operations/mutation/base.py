"""Mutation contract for the batched NEAT gene payload.

A mutation operator is a trace-time kernel over a whole population: the four payload
arrays enter and leave with identical static shapes, the population lives on axis 0 and
every gate is an elementwise mask — no Python loop over individuals, no ``vmap``, no
``scan``/``while_loop``/``cond``.
Structural keys are caller-allocated ``int32`` tensors, and the only randomness consumed
here is one rank-0 ``evoxc`` key, so a ``(seed, config, shape)`` triple reproduces a run.

The kernel is factored into two stages — :meth:`BaseMutation.mutate_structure` (at most
one structural change per individual) and :meth:`BaseMutation.mutate_values` (attribute
mutation of every existing gene).  Both are implemented here, because both are the same
for every operator family; a subclass only supplies the per-operation helpers
(``_mutate_add_node``, ``_mutate_delete_node``, ``_mutate_add_conn``, ``_mutate_delete_conn``)
and the four probabilities they gate on (:attr:`BaseMutation.structural_probabilities`).
"""

from typing import TYPE_CHECKING

import evoxc
import evoxc.numpy as enp

from ....common import ConfigError, State, StatefulBaseClass
from ...utils import GenomeTensors

if TYPE_CHECKING:  # pragma: no cover - typing only; keeps genome -> operations acyclic
    from ...base import BaseGenome

#: Number of structural operations the shared gate draw decides between, in the order
#: ``(node_add, node_delete, conn_add, conn_delete)``.  It doubles as the "no operation
#: won" index, which is why it is one past the last row.
_OPERATION_COUNT = 4


def _int32(value: int) -> evoxc.SymbolicTensor:
    """Rank-0 ``int32`` constant holding ``value``.

    Written out instead of a bare Python literal because an ``int32`` tensor combined
    with a Python ``int`` promotes to ``int64`` in ``evoxc`` (plain numpy result-type
    promotion), which would silently widen every index computation downstream.

    Args:
        value: Static integer baked into the trace.

    Returns:
        Rank-0 ``int32`` tensor inside the active trace.
    """
    return enp.full((), int(value), dtype=evoxc.int32)


def _broadcast_row(row: evoxc.SymbolicTensor, pop_size: int) -> evoxc.SymbolicTensor:
    """Spread one gene attribute row over the population axis.

    The gene factories (``new_identity_attrs``, ``new_zero_attrs``) build a single
    ``(W,)`` row because the values are static, while the payload writers take a
    ``(P, W)`` block; an elementwise add against a ``(1, W)`` view is the broadcasting
    form evoxc offers (there is no ``broadcast_to``).

    Args:
        row: ``(W,)`` attribute row of one gene.
        pop_size: Population size ``P`` (static).

    Returns:
        ``(P, W)`` tensor whose every row equals ``row``.
    """
    return enp.zeros((pop_size, 1), dtype=row.dtype) + enp.reshape(
        row, (1, row.shape[0])
    )


def _gated(
    gate: evoxc.SymbolicTensor,
    new: evoxc.SymbolicTensor,
    old: evoxc.SymbolicTensor,
) -> evoxc.SymbolicTensor:
    """Keep ``new`` where ``gate`` is true and ``old`` where it is false.

    The writers of ``genome.utils`` are unconditional (a scatter can never be a per-row
    no-op), so every structural helper writes garbage into a redirected row and this
    ``select`` is what turns "refused" into a bit-identical row — ``-0.0`` and ``NaN``
    included, because nothing is recomputed for a refused individual.

    Args:
        gate: ``(P,)`` bool eligibility mask.
        new: Written payload of rank 1, 2 or 3.
        old: Original payload of the same shape as ``new``.

    Returns:
        ``old`` where ``gate`` is false, ``new`` where it is true.
    """
    # ``select`` broadcasts the operands but not a lower-rank predicate, so the ``(P,)``
    # gate is expanded to the payload's rank (``(P, 1)``, ``(P, 1, 1)``, ...).
    mask = enp.reshape(gate, (gate.shape[0],) + (1,) * (len(old.shape) - 1))
    return evoxc.select(mask, new, old)


class BaseMutation(StatefulBaseClass):
    """Batched mutation operator: one structural change plus value mutation.

    Subclasses supply the two things that differ between operator families: the four
    probabilities :attr:`structural_probabilities` (validated by the subclass
    constructor) and the four per-operation helpers ``_mutate_add_node``,
    ``_mutate_delete_node``, ``_mutate_add_conn`` and ``_mutate_delete_conn``, each of
    which takes a trailing ``(P,)`` bool ``gate`` and is a batch-wide no-op where that
    gate is false.  Everything else — the key split, the shared gate draw, the operation
    ordering and the value mutation — lives here.

    Padding conventions are preserved throughout: absent ``int32`` key rows are ``I_INF``
    and absent float attribute rows are NaN, so no operator ever leaks a NaN into a key
    column or an ``I_INF`` into an attribute column.
    """

    def __call__(
        self,
        state: State,
        genome: "BaseGenome",
        key: evoxc.SymbolicTensor,
        node_keys: evoxc.SymbolicTensor,
        node_attrs: evoxc.SymbolicTensor,
        conn_keys: evoxc.SymbolicTensor,
        conn_attrs: evoxc.SymbolicTensor,
        new_node_keys: evoxc.SymbolicTensor,
        new_conn_keys: evoxc.SymbolicTensor,
    ) -> GenomeTensors:
        """Mutate a whole population in one kernel call.

        The key is split internally into one key per stage; :meth:`mutate_structure` runs
        first and :meth:`mutate_values` sees the arrays it produced.  Nothing is applied
        per individual: gates are elementwise masks on axis 0 and genes whose candidate is
        invalid are simply carried through.

        Args:
            state (State): Immutable host configuration container, read for gene metadata
                (attribute bounds, mutation rates); never mutated here.
            genome ("BaseGenome"): Genome owning the node/conn gene objects and the
                static invariants (``max_nodes``, ``max_conns``, ``max_in_degree``,
                ``network_type``).
            key (evoxc.SymbolicTensor): Single rank-0 ``evoxc`` RNG key for this call.
                ``evoxc`` keys are rank-0 only, so independence comes from batched draw
                shapes, not from splitting one key per individual.
            node_keys (evoxc.SymbolicTensor): ``(P, N) int32`` node key rows; ``I_INF``
                marks an absent row.  Input keys are rows ``0..num_inputs-1``, output keys
                are rows ``N-num_outputs..N-1`` and are never moved.
            node_attrs (evoxc.SymbolicTensor): ``(P, N, NA) float32`` node attribute rows;
                absent rows are NaN.
            conn_keys (evoxc.SymbolicTensor): ``(P, E, CK) int32`` connection keys —
                ``(in_key, out_key)`` plus ``historical_marker`` when the conn gene
                declares it; ``I_INF`` marks an absent row.
            conn_attrs (evoxc.SymbolicTensor): ``(P, E, CA) float32`` connection
                attributes (e.g. ``weight``); absent rows are NaN.
            new_node_keys (evoxc.SymbolicTensor): ``(P,) int32`` caller-allocated,
                population-unique keys for the node an add-node mutation may insert.
            new_conn_keys (evoxc.SymbolicTensor): ``(P, 3) int32`` caller-allocated,
                population-unique keys for the connections this generation may insert;
                column 0 feeds the add-node identity edge, column 1 the add-node copy edge,
                column 2 the add-conn edge.

        Returns:
            GenomeTensors: ``(node_keys, node_attrs, conn_keys, conn_attrs)`` with the shapes and
            dtypes of the inputs — the mutated population.

        Raises:
            ConfigError: If ``new_node_keys`` is not ``(P,)`` or ``new_conn_keys`` is
                not ``(P, 3)`` for the population size ``P`` of the payload arrays.
        """
        pop_size = node_keys.shape[0]
        if tuple(new_node_keys.shape) != (pop_size,):
            raise ConfigError(
                "new_node_keys must be (P,) for the payload's population size "
                f"{pop_size}, got {tuple(new_node_keys.shape)}"
            )
        if tuple(new_conn_keys.shape) != (pop_size, 3):
            raise ConfigError(
                "new_conn_keys must be (P, 3) for the payload's population size "
                f"{pop_size}, got {tuple(new_conn_keys.shape)}"
            )
        structure_key, values_key = evoxc.random.split_n(key, 2)
        node_keys, node_attrs, conn_keys, conn_attrs = self.mutate_structure(
            state,
            genome,
            structure_key,
            node_keys,
            node_attrs,
            conn_keys,
            conn_attrs,
            new_node_keys,
            new_conn_keys,
        )
        return self.mutate_values(
            state, genome, values_key, node_keys, node_attrs, conn_keys, conn_attrs
        )

    @property
    def structural_probabilities(self) -> tuple[float, float, float, float]:
        """Per-operation probabilities, in gate-draw order.

        The order is ``(node_add, node_delete, conn_add, conn_delete)`` and matches both
        the rows of the shared ``(4, P)`` draw and the order the helpers run in.  A
        subclass owns the four values as instance attributes — it is also the one that
        validates them, so a bad configuration fails at construction time.

        Returns:
            tuple[float, float, float, float]: The four probabilities, each in ``[0, 1]``.
        """
        return (self.node_add, self.node_delete, self.conn_add, self.conn_delete)

    def mutate_structure(
        self,
        state: State,
        genome: "BaseGenome",
        key: evoxc.SymbolicTensor,
        node_keys: evoxc.SymbolicTensor,
        node_attrs: evoxc.SymbolicTensor,
        conn_keys: evoxc.SymbolicTensor,
        conn_attrs: evoxc.SymbolicTensor,
        new_node_keys: evoxc.SymbolicTensor,
        new_conn_keys: evoxc.SymbolicTensor,
    ) -> GenomeTensors:
        """Apply at most one structural change per individual.

        One batched ``evoxc.random.uniform(key, (4, P))`` draw is compared against the
        four :attr:`structural_probabilities`; the *first* row that hit wins
        (``argmax`` plus a hit test, so row ``4`` means "nothing"), which is what keeps
        the change count at one per individual and per generation.  The four helpers then
        run in the order add-node, delete-node, add-conn, delete-conn, each with its own
        key (from one ``split_n``) and its own ``(P,)`` gate, so three of them are
        batch-wide no-ops by construction.

        Args:
            state (State): Immutable host configuration container.
            genome ("BaseGenome"): Genome providing gene objects and static capacity
                invariants.
            key (evoxc.SymbolicTensor): Rank-0 ``evoxc`` key; split into one key per
                operation plus one for the gate draw.
            node_keys (evoxc.SymbolicTensor): ``(P, N) int32`` node keys, ``I_INF`` padded.
            node_attrs (evoxc.SymbolicTensor): ``(P, N, NA) float32`` node attributes,
                NaN padded.
            conn_keys (evoxc.SymbolicTensor): ``(P, E, CK) int32`` connection keys,
                ``I_INF`` padded.
            conn_attrs (evoxc.SymbolicTensor): ``(P, E, CA) float32`` connection
                attributes, NaN padded.
            new_node_keys (evoxc.SymbolicTensor): ``(P,) int32`` allocated node keys.
            new_conn_keys (evoxc.SymbolicTensor): ``(P, 3) int32`` allocated connection
                keys.

        Returns:
            GenomeTensors: ``(node_keys, node_attrs, conn_keys, conn_attrs)`` after the structural
            stage; every individual whose gate is off is returned unchanged.

        Raises:
            ConfigError: If ``genome.network_type`` is neither ``"feedforward"`` nor
                ``"recurrent"`` — the add-connection helper would otherwise have to guess
                whether a cycle-closing edge is legal.
        """
        if genome.network_type not in ("feedforward", "recurrent"):
            raise ConfigError(
                "mutation needs genome.network_type to be 'feedforward' or 'recurrent', "
                f"got {genome.network_type!r}"
            )
        pop_size = node_keys.shape[0]
        # Four operation keys plus the gate key, in the order the helpers run.
        keys = evoxc.random.split_n(key, _OPERATION_COUNT + 1)
        gates = self._structural_gates(keys[_OPERATION_COUNT], pop_size)
        node_keys, node_attrs, conn_keys, conn_attrs = self._mutate_add_node(
            state,
            genome,
            keys[0],
            node_keys,
            node_attrs,
            conn_keys,
            conn_attrs,
            new_node_keys,
            new_conn_keys,
            gates[0],
        )
        node_keys, node_attrs, conn_keys, conn_attrs = self._mutate_delete_node(
            state,
            genome,
            keys[1],
            node_keys,
            node_attrs,
            conn_keys,
            conn_attrs,
            gates[1],
        )
        node_keys, node_attrs, conn_keys, conn_attrs = self._mutate_add_conn(
            state,
            genome,
            keys[2],
            node_keys,
            node_attrs,
            conn_keys,
            conn_attrs,
            new_conn_keys,
            gates[2],
        )
        node_keys, node_attrs, conn_keys, conn_attrs = self._mutate_delete_conn(
            state,
            genome,
            keys[3],
            node_keys,
            node_attrs,
            conn_keys,
            conn_attrs,
            gates[3],
        )
        return GenomeTensors(node_keys, node_attrs, conn_keys, conn_attrs)

    def mutate_values(
        self,
        state: State,
        genome: "BaseGenome",
        key: evoxc.SymbolicTensor,
        node_keys: evoxc.SymbolicTensor,
        node_attrs: evoxc.SymbolicTensor,
        conn_keys: evoxc.SymbolicTensor,
        conn_attrs: evoxc.SymbolicTensor,
    ) -> GenomeTensors:
        """Mutate the attributes of every existing gene in the population.

        One batched ``genome.node_gene.mutate`` and one ``genome.conn_gene.mutate`` call
        cover the whole ``(P, N, NA)`` / ``(P, E, CA)`` tables — the genes are batched over
        their leading axes, so no Python loop or per-individual key split is needed.  The
        result is re-masked with the padding pattern read *before* the mutation, so a
        padded row can never become finite.

        Only ``node_attrs`` and ``conn_attrs`` change; the key arrays are returned
        untouched, and padded rows keep their sentinel values (``I_INF`` keys, NaN attrs).

        Args:
            state (State): Immutable host configuration container.
            genome ("BaseGenome"): Genome whose ``node_gene``/``conn_gene`` perform the
                batched attribute mutation.
            key (evoxc.SymbolicTensor): Rank-0 ``evoxc`` key for this stage.
            node_keys (evoxc.SymbolicTensor): ``(P, N) int32`` node keys, ``I_INF`` padded.
            node_attrs (evoxc.SymbolicTensor): ``(P, N, NA) float32`` node attributes to
                mutate, NaN padded.
            conn_keys (evoxc.SymbolicTensor): ``(P, E, CK) int32`` connection keys,
                ``I_INF`` padded.
            conn_attrs (evoxc.SymbolicTensor): ``(P, E, CA) float32`` connection attributes
                to mutate, NaN padded.

        Returns:
            GenomeTensors: ``(node_keys, node_attrs, conn_keys, conn_attrs)`` with mutated
            attribute arrays and unchanged key arrays.
        """
        node_key, conn_key = evoxc.random.split_n(key, 2)
        node_padding = evoxc.isnan(node_attrs)
        conn_padding = evoxc.isnan(conn_attrs)
        mutated_nodes = genome.node_gene.mutate(state, node_key, node_attrs)
        mutated_conns = genome.conn_gene.mutate(state, conn_key, conn_attrs)
        return GenomeTensors(
            node_keys,
            evoxc.select(node_padding, node_attrs, mutated_nodes),
            conn_keys,
            evoxc.select(conn_padding, conn_attrs, mutated_conns),
        )

    def _structural_gates(
        self, key: evoxc.SymbolicTensor, pop_size: int
    ) -> tuple[
        evoxc.SymbolicTensor,
        evoxc.SymbolicTensor,
        evoxc.SymbolicTensor,
        evoxc.SymbolicTensor,
    ]:
        """Resolve which of the four structural operations each individual performs.

        The four probabilities are compared against one ``(4, P)`` uniform draw; the
        first row that hit wins, and an individual for which no row hit gets no structural
        change at all.  Both outcomes are ordinary elementwise results, so the kernel stays
        branch-free (``cond`` is a region op and cannot be batched).

        Args:
            key (evoxc.SymbolicTensor): Rank-0 ``evoxc`` key for this draw.
            pop_size (int): Population size ``P`` (static).

        Returns:
            tuple: four ``(P,)`` bool gates, in the order add-node, delete-node,
            add-conn, delete-conn; at most one of them is true per individual.

        Notes:
            The comparison runs against a ``(4, 1)`` column of static thresholds, so a
            probability of ``0`` produces an all-``False`` row (the operation can never
            win) and a probability of ``1`` an all-``True`` one, without a Python branch.
        """
        drawn = evoxc.random.uniform(
            key, (_OPERATION_COUNT, pop_size), 0.0, 1.0, dtype=evoxc.float32
        )
        thresholds = enp.reshape(
            evoxc.stack(
                [
                    enp.full((), float(probability), dtype=evoxc.float32)
                    for probability in self.structural_probabilities
                ],
                axis=0,
            ),
            (_OPERATION_COUNT, 1),
        )
        hit = evoxc.less(drawn, thresholds)
        # First hitting row, or the "no operation" index when the mask is all False
        # (`argmax` of an all-False row would otherwise name row 0, a real operation).
        winner = evoxc.select(
            evoxc.reduce_max(hit, axes=0),
            evoxc.cast(evoxc.argmax(hit, axis=0), evoxc.int32),
            _int32(_OPERATION_COUNT),
        )
        return tuple(
            evoxc.equal(winner, _int32(index)) for index in range(_OPERATION_COUNT)
        )


__all__ = ["BaseMutation"]
