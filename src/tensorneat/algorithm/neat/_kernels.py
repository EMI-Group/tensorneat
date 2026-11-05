"""Graph primitives and the guided species kernels behind :mod:`species`.

This private module holds the parts of the species algebra that are written
*against tensors* rather than against the :class:`SpeciesState` container:

* the graph-time constant/index helpers (``enp`` creation ops need a live trace,
  so nothing here may be precomputed on the host);
* the shared building blocks ``sort_desc_legacy``, ``prefix_sum_exclusive``,
  ``membership`` and ``scatter_row``;
* spawn allocation (:func:`spawn_numbers_by_rank` / ``_by_fitness`` on top of
  :func:`spawn_from_target`), block flattening (:func:`block_layout`), parent
  pairing (:func:`pair_parents`);
* the two fixed-length speciation scans (:func:`speciate`).

It deliberately imports nothing from :mod:`species`: the public controller wraps
these functions with its documented API, so the dependency points one way
(``species -> _kernels``) and there is no import cycle.  Everything here runs
inside an active ``evoxc`` trace and takes the static sizes it needs as explicit
arguments, because ``S``/``P`` are fixed at build time.
"""

from __future__ import annotations

from typing import Any, Callable

import numpy as np

import evoxc
import evoxc.numpy as enp

from tensorneat.common.errors import TensorNEATError
from tensorneat.common.tools import (
    I_INF,
    argmin_with_mask,
    fetch_first,
    flat_gather,
)

from ._distance import repeat_leading

#: Entry name under which the current population payload is stored.
POP_KEY = "pop"

#: Split-array spelling of the same payload (used when ``POP_KEY`` is absent).
SPLIT_POP_KEYS = ("pop_node_keys", "pop_node_attrs", "pop_conn_keys", "pop_conn_attrs")


# --------------------------------------------------------------------------- #
# graph-time constants and index vectors
# --------------------------------------------------------------------------- #


def const(value: float) -> Any:
    """0-d ``float32`` graph constant."""
    return enp.full((), value, dtype=evoxc.float32)


def int_const(value: int) -> Any:
    """0-d ``int32`` graph constant."""
    return enp.full((), value, dtype=evoxc.int32)


def nan() -> Any:
    """0-d ``float32`` ``NaN`` -- the "dead / absent" sentinel of float arrays."""
    return const(float("nan"))


def neg_inf() -> Any:
    """0-d ``float32`` ``-inf`` -- the "worse than anything" sentinel."""
    return const(float("-inf"))


def pos_inf() -> Any:
    """0-d ``float32`` ``+inf`` -- "not selectable" in a masked ``argmin``."""
    return const(float("inf"))


def iota_f32(length: int) -> Any:
    """``(length,)`` ``float32`` index vector ``[0, 1, ...]``."""
    return enp.arange(length, dtype=evoxc.float32)


def iota_i32(length: int) -> Any:
    """``(length,)`` ``int32`` index vector ``[0, 1, ...]``; the scan ``xs``."""
    return enp.arange(length, dtype=evoxc.int32)


def static_i32(values: Any) -> Any:
    """Bake a static ``int32`` table (index vector, permutation) as a constant.

    Fully static data belongs in a graph constant, not in traced arithmetic:
    ``evoxc.constant`` emits a single ``constant`` op, while the equivalent
    ``enp.full``/``subtract``/``enp.tril``/``evoxc.flip`` spelling would emit
    ops that the StableHLO v1 exporter either grows or rejects outright
    (``tril`` and ``flip`` are *deferred* there).  The values are exact
    integers, so the constant is bit-identical to the computed form.
    """
    return evoxc.constant(evoxc.from_numpy(np.asarray(values, dtype=np.int32)))


def static_f32(values: Any) -> Any:
    """Bake a static ``float32`` table (mask, gate matrix) as a constant.

    See :func:`static_i32`; the entries here are exact ``0.0``/``1.0``.
    """
    array = np.ascontiguousarray(values, dtype=np.float32)
    return evoxc.constant(evoxc.from_numpy(array))


def reversed_index(length: int) -> Any:
    """``(length,)`` ``int32`` index vector ``[length - 1, ..., 1, 0]``.

    The exportable replacement for ``evoxc.flip(x, 0)``, which the StableHLO
    v1 exporter does not map: ``gather(x, reversed_index(n), axis=0)`` is the
    same reversal, since a single-axis gather with static indices is
    supported.
    """
    return static_i32(np.arange(length - 1, -1, -1))


def scalar_f32(value: Any) -> Any:
    """Normalise a traced value to a rank-0 ``float32``.

    A 0-d tensor handed to a graph as an *input* can arrive with shape ``(1,)``,
    so a scalar that is later broadcast against ``(S,)`` fields is reshaped here
    instead of at every use site.
    """
    return evoxc.reshape(evoxc.cast(evoxc.reshape(value, (1,)), evoxc.float32), ())


def payload_like(
    template: Any, node_keys: Any, node_attrs: Any, conn_keys: Any, conn_attrs: Any
) -> Any:
    """Rebuild ``template``'s payload type from the four split arrays.

    ``distance_fn`` receives a *population-shaped* payload on both sides -- one
    row per (centre, individual) pair (see :mod:`._distance`) -- so a per-slot
    centre row is tiled to ``pop_size`` rows before it is wrapped in the caller's
    own payload class (``genome.utils.GenomeTensors``) instead of a second,
    locally defined copy; importing ``tensorneat.genome`` from here would make
    ``algorithm`` depend on a sibling node at package import time.
    """
    return type(template)(node_keys, node_attrs, conn_keys, conn_attrs)


# --------------------------------------------------------------------------- #
# shared building blocks
# --------------------------------------------------------------------------- #


def sort_desc_legacy(x: Any, axis: int = -1) -> Any:
    """Descending sort indices that reproduce the legacy tie order.

    Stable ascending ``argsort`` followed by a gather with the reversal
    permutation ``n - 1 - arange(n)``; ``argsort(descending=True)`` breaks ties
    the other way and would change which species becomes slot ``0``.
    """
    rank = len(x.shape)
    resolved = axis + rank if axis < 0 else axis
    order = evoxc.argsort(x, axis=resolved, stable=True)
    length = x.shape[resolved]
    reversal = evoxc.subtract(
        enp.full((length,), length - 1, dtype=evoxc.int64),
        enp.arange(length, dtype=evoxc.int64),
    )
    return evoxc.gather(order, reversal, axis=resolved)


def prefix_sum_exclusive(values: Any) -> Any:
    """Exclusive prefix sum of a ``(S,)`` float vector via a triangular ``dot``.

    ``evoxc.cumsum`` exists but is a *deferred* op on the compiler backends, and
    so is ``enp.tril``; the strictly-lower-triangular gate matrix is therefore
    baked as a static ``float32`` constant (``length`` is static at build time
    and the entries are exact ``0.0``/``1.0``), which is both exportable and
    one op cheaper than the traced ``tril``.
    """
    length = values.shape[0]
    triangular = static_f32(np.tril(np.ones((length, length), dtype=np.float32), -1))
    return evoxc.reshape(evoxc.dot(triangular, enp.expand_dims(values, 1)), (length,))


def scatter_row(table: Any, slot: Any, update: Any) -> Any:
    """Replace row ``slot`` of a ``(S, ...)`` table with ``update``.

    ``evoxc.scatter`` writes the cells numpy's ``x[slot] = update`` writes, but
    the StableHLO v1 exporter only supports ``scatter`` on a rank-1 ``axis=0``
    operand and on rank-2 row/column writes -- a rank-3 gene table
    (``(S, N, NA)`` node attrs, ``(S, E, CK)`` conn keys, ...) is rejected
    outright.  Flattening the trailing axes into one row length ``R`` turns the
    same write into a rank-2 ``axis=0`` row scatter, which exports; the
    reshape round-trip is shape-only, so the written cells are identical.
    """
    shape = tuple(table.shape)
    if len(shape) == 2:
        return evoxc.scatter(table, slot, update)
    row_len = 1
    for dim in shape[1:]:
        row_len *= dim
    flat = evoxc.reshape(table, (shape[0], row_len))
    written = evoxc.scatter(flat, slot, evoxc.reshape(update, (row_len,)))
    return evoxc.reshape(written, shape)


def first_centre_table(species_size: int, row: Any, fill: Any, dtype: Any) -> Any:
    """``(S,) + row.shape`` centre table whose row ``0`` is ``row``, rest ``fill``.

    Generation 0 has exactly one live species, whose centre is the first
    individual, so every trailing row is the genome's own dead-slot padding
    (``0`` for the integer key arrays, ``NaN`` for the float attribute arrays).

    Written as a ``concatenate`` of the reshaped first row with a constant fill
    block rather than as a row ``scatter``: every row is written here, so there
    is nothing to mask, and the StableHLO v1 exporter rejects a ``scatter`` on a
    rank-3 operand along axis 0 (the ``(S, N, NA)``/``(S, E, CA)`` tables) while
    a ``concatenate`` exports at every rank.

    Args:
        species_size (int): Static species slots ``S`` (``>= 1``).
        row: One individual's gene array, rank ``1`` (``node_keys``) or ``2``
            (attributes / conn keys), without a species axis.
        fill: Padding value for the dead rows.
        dtype: Static dtype of the table (``evoxc.int32`` or ``evoxc.float32``).

    Returns:
        ``(S,) + row.shape`` tensor, row ``0`` equal to ``row``.
    """
    head = evoxc.reshape(row, (1,) + tuple(row.shape))
    if species_size == 1:
        return head
    tail = enp.full((species_size - 1,) + tuple(row.shape), fill, dtype=dtype)
    return enp.concatenate([head, tail], 0)

def membership(species_keys: Any, idx2species: Any) -> Any:
    """``(S, P)`` mask ``member[i, p] = idx2species[p] == species_keys[i]``.

    ``NaN`` never compares equal, so a dead slot and an unassigned genome both
    match nothing.
    """
    return evoxc.equal(
        enp.expand_dims(species_keys, 1), enp.expand_dims(idx2species, 0)
    )


# --------------------------------------------------------------------------- #
# spawn allocation
# --------------------------------------------------------------------------- #


def spawn_from_target(
    species_keys: Any,
    member_count: Any,
    target: Any,
    pop_size: int,
    min_species_size: int,
    change_rate: float,
) -> Any:
    """Damp, floor, mask and integerise a spawn target into a spawn budget.

    ``damped = previous + (target - previous) * change_rate`` with ``previous =
    member_count`` (NaN treated as ``0``), floored at ``min_species_size``,
    forced to ``0.0`` on dead slots, floored to int32, and finally corrected so
    the total is exactly ``pop_size``.

    The residual correction is a single ``scatter`` on the **first live slot**
    (``evoxc.scatter`` is replacement-only), which after the descending species
    sort of :meth:`SpeciesController.update_species` is slot ``0`` -- so the
    common path is unchanged -- while a hand-built or degenerate table can never
    hand population slots to a dead species.  The residual is non-negative in
    practice: ``sum(member_count)`` and ``sum(target)`` are both bounded by
    ``pop_size``, and the ``min_species_size`` floor cannot push the total past
    ``pop_size`` either, because ``species_size * min_species_size <= pop_size``
    is validated in the controller constructor.
    """
    live = evoxc.logical_not(evoxc.isnan(species_keys))
    previous = evoxc.select(evoxc.isnan(member_count), const(0.0), member_count)
    damped = evoxc.add(
        previous,
        evoxc.multiply(evoxc.subtract(target, previous), float(change_rate)),
    )
    damped = evoxc.select(evoxc.isnan(damped), const(0.0), damped)
    damped = evoxc.maximum(damped, float(min_species_size))
    damped = evoxc.select(live, damped, const(0.0))
    spawn = evoxc.cast(evoxc.floor(damped), evoxc.int32)
    total = evoxc.cast(evoxc.sum(spawn, axes=0), evoxc.int32)
    residual = evoxc.subtract(pop_size, total)
    # First live slot: ``argmax`` of the 0/1 liveness mask, and index ``0`` when
    # the whole table is dead (nothing left to protect in that case).
    head_slot = evoxc.cast(evoxc.argmax(evoxc.cast(live, evoxc.int32), axis=0), evoxc.int32)
    head = evoxc.cast(evoxc.add(evoxc.gather(spawn, head_slot, axis=0), residual), evoxc.int32)
    return evoxc.scatter(spawn, head_slot, head)


def spawn_numbers_by_rank(
    species_keys: Any,
    member_count: Any,
    pop_size: int,
    species_size: int,
    min_species_size: int,
    change_rate: float,
) -> Any:
    """Linear-ranking spawn allocation over the *live* slots of a sorted table.

    ``n = sum(~isnan(species_keys))``, rank score ``n - i``, denominator the
    triangular number ``(n + 1) * n / 2`` floored at ``1.0`` (the legacy code
    divided by zero on an all-dead table), target
    ``floor(rank_score / denominator * pop_size)``.
    """
    live = evoxc.logical_not(evoxc.isnan(species_keys))
    live_count = evoxc.sum(evoxc.cast(live, evoxc.float32), axes=0)
    rank_score = evoxc.subtract(live_count, iota_f32(species_size))
    denominator = evoxc.maximum(
        evoxc.multiply(evoxc.multiply(evoxc.add(live_count, 1.0), live_count), 0.5),
        const(1.0),
    )
    target = evoxc.floor(
        evoxc.multiply(evoxc.divide(rank_score, denominator), float(pop_size))
    )
    return spawn_from_target(
        species_keys, member_count, target, pop_size, min_species_size, change_rate
    )


def spawn_numbers_by_fitness(
    species_keys: Any,
    best_fitness: Any,
    member_count: Any,
    pop_size: int,
    min_species_size: int,
    change_rate: float,
) -> Any:
    """Fitness-proportional spawn allocation (``"fitness"`` mode).

    ``f' = f - min_live(f) + 1`` so the worst live species still gets a positive
    weight, rate ``f' / sum_live(f')``, target ``floor(rate * pop_size)``.
    Every reduction masks dead slots first: the NaN ``best_fitness`` of a dead
    slot must not poison ``min``/``sum``, and a freshly created slot's ``-inf``
    best fitness is cleaned to weight ``0.0`` so it is carried by the
    ``min_species_size`` floor alone.
    """
    live = evoxc.logical_not(evoxc.isnan(species_keys))
    masked = evoxc.select(live, best_fitness, pos_inf())
    lowest = evoxc.reduce_min(masked, axes=0)
    shifted = evoxc.add(evoxc.subtract(best_fitness, lowest), 1.0)
    cleaned = evoxc.nan_to_num(shifted, nan=0.0, posinf=0.0, neginf=0.0)
    weight = evoxc.select(live, cleaned, const(0.0))
    total = evoxc.maximum(evoxc.sum(weight, axes=0), const(1.0))
    target = evoxc.floor(
        evoxc.multiply(evoxc.divide(weight, total), float(pop_size))
    )
    return spawn_from_target(
        species_keys, member_count, target, pop_size, min_species_size, change_rate
    )


# --------------------------------------------------------------------------- #
# block flattening and parent pairing
# --------------------------------------------------------------------------- #


def block_layout(spawn_number: Any, pop_size: int, species_size: int) -> Any:
    """Map each population slot to its species block: ``(loc, idx_in_block)``.

    With the exclusive prefix sum ``E`` of ``spawn_number`` and
    ``K = sum(spawn) == pop_size``, slot ``p`` belongs to

    ``loc(p)         = the last ``i`` with ``spawn[i] > 0`` and ``E[i] <= p````
    ``idx_in_block(p) = p - E[loc(p)]``

    Counting only blocks with ``spawn > 0`` is what makes a zero-spawn (dead or
    empty) species consume no population slot.  ``loc`` must be the *last*
    qualifying block rather than the number of qualifying blocks: zero-spawn
    species can sit in the middle of the table (after the sort only dead slots
    are guaranteed to trail), so a count would point at the wrong block -- hence
    the reversed ``argmax`` (``argmax`` returns the first maximum), with the
    reversal spelled as a static-index ``gather`` (:func:`reversed_index`)
    because ``evoxc.flip`` is a deferred op on the StableHLO exporter.
    """
    spawn = evoxc.cast(spawn_number, evoxc.int32)
    exclusive = prefix_sum_exclusive(evoxc.cast(spawn, evoxc.float32))
    slots = iota_f32(pop_size)
    mask = evoxc.logical_and(
        enp.expand_dims(evoxc.greater(spawn, int_const(0)), 1),
        evoxc.less_equal(enp.expand_dims(exclusive, 1), enp.expand_dims(slots, 0)),
    )
    reversed_loc = evoxc.cast(
        evoxc.argmax(
            evoxc.gather(
                evoxc.cast(mask, evoxc.int32), reversed_index(species_size), axis=0
            ),
            axis=0,
        ),
        evoxc.int32,
    )
    loc = evoxc.subtract(int_const(species_size - 1), reversed_loc)
    idx_in_block = evoxc.cast(
        evoxc.subtract(slots, evoxc.gather(exclusive, loc, axis=0)), evoxc.int32
    )
    return loc, idx_in_block


def pair_parents(
    species_keys: Any,
    idx2species: Any,
    member_count: Any,
    selection_key: Any,
    spawn_number: Any,
    fitness: Any,
    pop_size: int,
    species_size: int,
    survival_threshold: float,
    genome_elitism: int,
) -> Any:
    """Draw two parents per population slot and decide winner/loser.

    Members are ranked by fitness descending (``sort_desc_legacy`` over
    ``select(membership, fitness, -inf)`` applied to the population index
    vector), the survivor count is ``clamp(floor(survival_threshold *
    member_count), 1, min(member_count, pop_size))`` and the first
    ``survive_size`` ranks are the candidate parents.

    ``survive_size`` is floored at ``1`` *before* it becomes the ``randint``
    bound, which is the fix for the legacy zero-survivor division: a one-member
    species draws rank ``0`` twice, i.e. two identical parents, and
    self-crossover is the identity.  The extra ``pop_size`` cap is a no-op for a
    valid table and keeps the bound inside the ``(S, P)`` ranking width.

    The winner decision is taken on the *species-space* draws (ties go to the
    father, ``>=``, matching the legacy comparison) before elite pinning, so it
    does not depend on which population slot ends up owning the pair.

    Elite slot ``k`` of a block (``idx_in_block == k < genome_elitism``) pins both
    parents to that block's ``k``-th ranked member (the legacy elementwise pin), so
    the species' top ``genome_elitism`` **distinct** members survive unmutated; the
    rank is clamped at ``live_members - 1`` so an elite never adopts a non-member or
    a sanitised ``-inf`` genome.
    """
    members = membership(species_keys, idx2species)
    masked = evoxc.select(members, enp.expand_dims(fitness, 0), neg_inf())
    ranking = evoxc.cast(sort_desc_legacy(masked, axis=1), evoxc.int32)
    previous = evoxc.select(evoxc.isnan(member_count), const(0.0), member_count)
    floored = evoxc.floor(evoxc.multiply(previous, float(survival_threshold)))
    ceiling = evoxc.maximum(evoxc.minimum(previous, const(float(pop_size))), const(1.0))
    survive = evoxc.cast(evoxc.clamp(floored, const(1.0), ceiling), evoxc.int32)
    draws = evoxc.random.randint(
        selection_key,
        (species_size, 2, pop_size),
        0,
        evoxc.reshape(survive, (species_size, 1, 1)),
        dtype=evoxc.int32,
    )
    # ``evoxc.gather`` replaces its axis instead of indexing per row, so the
    # per-row draw must go through the flattened ``common.tools.flat_gather``.
    parent = flat_gather(ranking, draws)
    width = species_size * pop_size
    father = evoxc.reshape(evoxc.gather(parent, 0, axis=1), (width,))
    mother = evoxc.reshape(evoxc.gather(parent, 1, axis=1), (width,))
    father_wins = evoxc.greater_equal(
        evoxc.gather(fitness, father, axis=0), evoxc.gather(fitness, mother, axis=0)
    )
    winner_species = evoxc.select(father_wins, father, mother)
    loser_species = evoxc.select(father_wins, mother, father)

    loc, idx_in_block = block_layout(spawn_number, pop_size, species_size)
    flat_slot = evoxc.add(evoxc.multiply(loc, int_const(pop_size)), idx_in_block)
    winner = evoxc.gather(winner_species, flat_slot, axis=0)
    loser = evoxc.gather(loser_species, flat_slot, axis=0)
    elite_mask = evoxc.less(idx_in_block, genome_elitism)
    # Elite slot ``c`` pins rank ``min(c, live_members - 1)``: the ranking is descending
    # and every finite fitness beats the ``-inf`` a sanitised genome maps to, so the
    # finite members head the row and the clamp stays on a live member.
    alive = evoxc.cast(evoxc.greater(fitness, neg_inf()), evoxc.float32)
    live = evoxc.sum(evoxc.select(members, enp.expand_dims(alive, 0), const(0.0)), axes=1)
    cap = evoxc.maximum(evoxc.subtract(evoxc.cast(live, evoxc.int32), int_const(1)), int_const(0))
    ranks = enp.reshape(enp.arange(pop_size, dtype=evoxc.int32), (1, pop_size))
    block_rank = evoxc.minimum(ranks, enp.expand_dims(cap, 1))
    pinned = evoxc.gather(evoxc.reshape(flat_gather(ranking, block_rank), (width,)), flat_slot, axis=0)
    return (
        evoxc.select(elite_mask, pinned, winner),
        evoxc.select(elite_mask, pinned, loser),
        elite_mask,
    )


# --------------------------------------------------------------------------- #
# speciation
# --------------------------------------------------------------------------- #


def read_population(state: Any, pop: Any = None) -> Any:
    """Read the current population payload handed to ``distance_fn``.

    An explicit ``pop`` wins; otherwise the ``POP_KEY`` entry is used, then the
    four ``SPLIT_POP_KEYS`` arrays.  There is no eager or packed fallback: a
    missing payload is a configuration error, never a silently empty population.
    """
    if pop is not None:
        return pop
    if POP_KEY in state:
        return getattr(state, POP_KEY)
    if all(name in state for name in SPLIT_POP_KEYS):
        # Deferred: ``tensorneat.genome`` is a sibling node, and importing it at
        # module scope would make ``algorithm`` depend on it while the package
        # itself is still initialising.
        from tensorneat.genome.utils import GenomeTensors

        return GenomeTensors(*(getattr(state, name) for name in SPLIT_POP_KEYS))
    raise TensorNEATError(
        "speciate needs the current population payload: register it as "
        f"'{POP_KEY}' (the four split arrays of a GenomeTensors) or as the "
        f"entries {SPLIT_POP_KEYS}, or pass it explicitly."
    )


#: Largest ``species_size * pop_size`` species table that keeps the lazy
#: per-slot ``evoxc.cond`` distance rows; above it the unconditional per-slot
#: call of the pre-gate implementation runs instead.
#:
#: The lazy ``cond`` only pays off while the species table is *sparse*, because a
#: skipped slot still leaves the two ``if`` regions (and their carries) in the
#: graph: it pays for the compatibility passes it cannot use, but carries the
#: region overhead of every slot.  The table is sparse only on small populations
#: -- 1-3 live slots out of 20 at pop 8/100, 20 of 20 at pop >= 500.
#: Measured with one configuration per fresh process, ``Pipeline.step`` median of
#: one warmup plus five timed steps after five setup steps (build excluded), two
#: runs per configuration alternating the ungated tree and the gated tree:
#:
#: * pop 100 (3 live): 252.8 / 251.6 ms ungated vs 111.9 / 111.3 ms gated -- a
#:   2.3x whole-generation win;
#: * pop 200 (20 live): 454.3 / 485.0 ms vs 421.2 / 417.1 ms -- parity (the gated
#:   build is 8-10 % ahead on this box, i.e. within run-to-run spread);
#: * pop 300 (15 live): 553.5 / 558.4 ms vs 444.2 / 458.5 ms -- a 1.2x win;
#: * pop 500 (20 live): 1101.7 / 1058.3 ms vs 1099.3 / 1074.0 ms -- parity, while
#:   the *unconditional* ``cond`` build costs 1024.6 / 1029.6 ms against 801.8 /
#:   816.7 ms ungated on the measurement box, a 26 % loss that lives in the
#:   in-graph cost of the nested ``if`` regions and not in extra passes
#:   (``tell`` 647.8 ms ungated vs 654.1 ms gated);
#: * pop 1000 (20 live): 1976.4 / 2007.3 ms vs 2030.1 / 2048.5 ms -- parity
#:   (+2.4 %, noise between two histories of the same graph).
#:
#: Hence a gate rather than an unconditional ``cond``: the crossover sits between
#: the pop-300 and pop-500 tables, i.e. ``6000`` rows (``20 * 300``) is gated and
#: ``10 000`` (``20 * 500``) is not.  ``GATED_ROWS`` is only ever compared against
#: static Python ints, so the choice is made at trace time and costs no op.
GATED_ROWS = 6000


def gated(species_size: int, pop_size: int) -> bool:
    """Whether ``speciate`` computes its distance rows lazily behind a ``cond``.

    Args:
        species_size (int): Static number of species slots ``S``.
        pop_size (int): Static population size ``P``.

    Returns:
        bool: ``True`` when ``species_size * pop_size <= GATED_ROWS``, i.e. when
            the species table is sparse enough that skipping dead/finished slots
            buys more than the nested-``if`` regions cost (see ``GATED_ROWS``).

    Note:
        A pure host-side predicate on static ints: the caller branches on it with
        an ordinary Python ``if`` at trace time, never with a traced ``cond``.
    """
    return species_size * pop_size <= GATED_ROWS


def speciate(
    state: Any,
    species_state: Any,
    distance_fn: Callable,
    population: Any,
    pop_size: int,
    species_size: int,
    compatibility_threshold: float,
    gate: bool,
) -> tuple:
    """Reassign the population to species with two fixed-length scans.

    Returns the ten :class:`SpeciesState` field values, in field order:
    ``(species_keys, best_fitness, last_improved, member_count, idx2species,
    center_node_keys, center_node_attrs, center_conn_keys, center_conn_attrs,
    next_species_key)``.

    Scan 1 (centre reservation) claims, per live slot, the nearest still
    unassigned genome and re-elects that genome as the slot's centre; a dead slot
    writes nothing.  Scan 2 (member assignment) assigns every genome within
    ``compatibility_threshold`` that is unassigned or closer to this centre than
    to its current one, and materialises a new species (keyed
    ``next_species_key``) on a dead slot when anything is still unassigned.  An
    iteration with nothing to do performs no write, which reproduces the legacy
    early-exit ``while_loop`` on a static trip count.

    **Carried state.**  Scan 1 carries the four centre gene tables (it is the
    pass that re-elects centres); scan 2 carries two rank-1 per-slot tables
    instead -- ``seed_table (S,) int32`` and ``created_mask (S,) float32`` -- and
    the four centre tables are rebuilt once after it.  Scan 2's own centre writes
    were the identity on every non-creating slot, so only a created row
    (``pop[seed_table[slot]]``) ever differs from the scan-1 table; recording
    that pair replaces four row scatters per iteration and drops the four gene
    tables out of the ``while`` carry without changing a single value.  Both
    scans carry ``assign``/``dist``/``keys_ext`` and nothing else.

    **Distance rows.**  ``distance_fn`` is the digest-backed callable built by
    ``NEAT.tell`` from ``genome.prepare_distance`` (see
    :class:`._distance.DistanceRows`).  Every call it receives carries a
    *population-shaped* centre payload -- ``pop_size`` rows, row ``p`` measured
    against population member ``p`` -- and returns one ``(pop_size,)`` row, so a
    phase's rows are one such call per slot, never one
    ``species_size * pop_size`` batch.  Which form a phase uses is a static
    choice on ``gate``:

    * **Sparse table** (``gate`` true, ``species_size * pop_size <=
      GATED_ROWS``): rows are computed lazily, one slot at a time, behind a
      single ``evoxc.cond`` per scan iteration, the predicate being exactly the
      mask that lets that iteration write -- scan 1 ``live and any_pending``, scan
      2 ``live or create``.  A table with 3 live slots out of 20 would waste 17 of
      the 20 rows of the table, so the lazy path is kept below the gate.  A
      skipped iteration reads a zero row, which its own write mask (``claim`` in
      scan 1, ``take`` in scan 2) discards, so the substitution is bit-identical:
      no reader ever consumes the row of a slot that is not allowed to write.
    * **Dense table** (``gate`` false): scan 1's whole centre table is measured
      *before* the scan through ``distance_fn.matrix`` -- one population-shaped
      call per centre row -- and each iteration reads its own row with a
      ``gather``.  Above ``GATED_ROWS`` the species table is full, so a skipped
      slot's ``if`` region would cost more than the row it skips (see
      ``GATED_ROWS``), while measuring the rows up front keeps the compatibility
      kernel out of the reservation ``while`` body entirely.  ``matrix`` runs
      those calls inside its own fixed-length ``scan``, one centre per
      iteration, so each iteration's region environment -- the kernel's
      ``(pop_size, 2 * max_conns, ...)`` transients -- is released before the
      next centre is measured, where a top-level loop would keep all
      ``species_size`` of them alive for the rest of the generation.  Scan 1's
      rows are the ones the gated path would read: slot ``i``'s centre is the
      carry row ``i``, and no earlier iteration can rewrite it (every
      ``scatter`` writes index ``slot`` and each slot is visited once).  Scan 2
      still measures its selected centre per iteration (``pop[seed]`` for a
      creating slot, the scan-1 centre row otherwise -- a closed-over value,
      because scan 2 no longer carries the centre tables), because a creating
      slot's centre is data-dependent.
      This form needs a callable with ``matrix`` (:class:`._distance.DistanceRows`);
      any other callable raises ``TensorNEATError`` instead of falling back to the
      slower shape, and the gated path keeps accepting an arbitrary callable.

    ``gate`` is a static Python ``bool``, so exactly one of the two paths is
    traced.  Bit-identity holds across the two paths by construction -- both
    compute the same ``(pop_size,)`` rows from the same kernel with the same
    population-shaped centre.
    """
    keys_ext = enp.concatenate(
        [species_state.species_keys, enp.reshape(species_state.next_species_key, (1,))],
        0,
    )
    centre_fields = (
        species_state.center_node_keys,
        species_state.center_node_attrs,
        species_state.center_conn_keys,
        species_state.center_conn_attrs,
    )
    carry = (
        enp.full((pop_size,), float("nan"), dtype=evoxc.float32),
        enp.full((pop_size,), float("inf"), dtype=evoxc.float32),
        keys_ext,
        *centre_fields,
    )
    population_fields = tuple(population)
    centre_payload = payload_like(population, *centre_fields)
    # Trace-time constants are built once here and closed over by the scan
    # bodies: a ``while`` body may reference enclosing SSA values, so one
    # ``constant`` op at the top level replaces the one every body would
    # otherwise re-emit on each of its ``species_size`` iterations.  The values
    # and their dtypes are exactly what the bodies used to build in place.
    iinf = int_const(I_INF)
    zero_i32 = int_const(0)
    zero_f32 = const(0.0)
    pop_index = iota_i32(pop_size)
    threshold = const(float(compatibility_threshold))
    skipped_row = enp.zeros((pop_size,), dtype=evoxc.float32)
    # The dense path measures whole centre rows *outside* the scans, so the
    # callable must expose the by-name table form; raised here rather than
    # silently falling back to the per-iteration shape this path avoids.
    if not gate and not hasattr(distance_fn, "matrix"):
        raise TensorNEATError(
            "speciate: the dense distance table (species_size * pop_size > "
            f"{GATED_ROWS}) needs a digest-backed distance_fn with a matrix "
            f"method, got {type(distance_fn).__name__}; only the lazy gated path "
            "accepts an arbitrary callable"
        )

    def single_payload(rows):
        """Wrap four 1-D centre rows as a population-shaped payload.

        The genome's distance kernel takes a population-shaped centre -- one row
        per (centre, individual) pair -- so the slot's single centre row is
        repeated ``pop_size`` times with :func:`._distance.repeat_leading`: every
        row of the payload is the same centre, which is what makes the returned
        ``(pop_size,)`` vector the centre's distance row.  A bare row would be
        read as a rank-1 payload.
        """
        return payload_like(
            population,
            *(
                repeat_leading(evoxc.reshape(row, (1,) + tuple(row.shape)), pop_size)
                for row in rows
            ),
        )

    def centre_rows(fields, slot):
        """Gather the centre row held in ``slot`` of each of the four ``fields``.

        The result is the input of :func:`single_payload`: a tuple of 1-D rows,
        one per split gene array.
        """
        return tuple(evoxc.gather(field, slot, axis=0) for field in fields)

    def selected_centre(create, seed, slot, carry_centre):
        """Centre rows of a scan-2 slot, chosen before the ``cond``.

        ``pop[seed]`` for a slot that is creating a species, the scan-1 centre
        row otherwise; the create-marking scatters leave those rows untouched
        when ``create`` is false, so both spellings agree.
        """
        return tuple(
            evoxc.select(
                create,
                evoxc.gather(field, seed, axis=0),
                evoxc.gather(centre_field, slot, axis=0),
            )
            for field, centre_field in zip(population_fields, carry_centre)
        )

    def gated_row(active, compute_row, operands):
        """One ``(pop_size,)`` compatibility row, computed only when ``active``.

        ``compute_row`` is the single true-branch body of the ``if``; the false
        branch returns the shared zero row (``skipped_row``), which every
        caller's write mask discards.

        This is the one nested-control-flow site of the subtree: the ``cond`` is
        built *inside* a ``scan`` step body.  Both branches are statically built,
        ``active`` is a traced mask (never a host value), and the nested form is
        proven to compile on ``iree``-llvm-cpu, with the numpy backend as the
        reference; ``vmap`` is still never applied over control flow.
        """

        def zero_row(*_skipped):
            """Row of a skipped slot: zeros, never read."""
            return skipped_row

        return evoxc.cond(active, compute_row, zero_row, *operands)

    def reserve(carry, slot):
        """Scan 1: claim the nearest unassigned genome for one live slot."""
        assign, dist, keys_ext, node_keys, node_attrs, conn_keys, conn_attrs = carry
        key_slot = evoxc.gather(keys_ext, slot, axis=0)
        live = evoxc.logical_not(evoxc.isnan(key_slot))
        pending = evoxc.isnan(assign)
        any_pending = evoxc.reduce_max(pending, axes=0)
        claim = evoxc.logical_and(live, any_pending)

        def compute_row(
            centre_node_keys,
            centre_node_attrs,
            centre_conn_keys,
            centre_conn_attrs,
            pop_node_keys,
            pop_node_attrs,
            pop_conn_keys,
            pop_conn_attrs,
            centre_slot,
        ):
            """Distance row of the carry centre ``centre_slot``, ``(P,)`` floats.

            The payload is built *in here* from the four carry arrays, so a
            skipped slot pays neither the gather nor the compatibility kernel.
            """
            return distance_fn(
                state,
                single_payload(
                    centre_rows(
                        (
                            centre_node_keys,
                            centre_node_attrs,
                            centre_conn_keys,
                            centre_conn_attrs,
                        ),
                        centre_slot,
                    )
                ),
                payload_like(
                    population,
                    pop_node_keys,
                    pop_node_attrs,
                    pop_conn_keys,
                    pop_conn_attrs,
                ),
            )

        operands = (node_keys, node_attrs, conn_keys, conn_attrs) + population_fields
        if gate:
            distances = gated_row(claim, compute_row, operands + (slot,))
        else:
            distances = evoxc.gather(rows1, slot, axis=0)
        # ``argmin_with_mask`` folds "nothing left to claim" into the ``I_INF``
        # sentinel index, which is clamped to row ``0`` before it is used.
        closest = evoxc.reshape(
            argmin_with_mask(
                enp.expand_dims(distances, 0), enp.expand_dims(pending, 0)
            ),
            (),
        )
        closest = evoxc.select(evoxc.equal(closest, iinf), zero_i32, closest)
        onehot = evoxc.logical_and(evoxc.equal(pop_index, closest), claim)
        assign = evoxc.select(onehot, key_slot, assign)
        dist = evoxc.select(onehot, zero_f32, dist)
        node_keys = scatter_row(
            node_keys,
            slot,
            evoxc.select(
                claim,
                evoxc.gather(population.node_keys, closest, axis=0),
                evoxc.gather(node_keys, slot, axis=0),
            ),
        )
        node_attrs = scatter_row(
            node_attrs,
            slot,
            evoxc.select(
                claim,
                evoxc.gather(population.node_attrs, closest, axis=0),
                evoxc.gather(node_attrs, slot, axis=0),
            ),
        )
        conn_keys = scatter_row(
            conn_keys,
            slot,
            evoxc.select(
                claim,
                evoxc.gather(population.conn_keys, closest, axis=0),
                evoxc.gather(conn_keys, slot, axis=0),
            ),
        )
        conn_attrs = scatter_row(
            conn_attrs,
            slot,
            evoxc.select(
                claim,
                evoxc.gather(population.conn_attrs, closest, axis=0),
                evoxc.gather(conn_attrs, slot, axis=0),
            ),
        )
        return (
            assign,
            dist,
            keys_ext,
            node_keys,
            node_attrs,
            conn_keys,
            conn_attrs,
        ), assign

    # Scan 2 carries two rank-1 per-slot tables instead of the four centre gene
    # tables: its centre writes were the identity on every non-creating slot, so
    # recording ``seed``/``create`` (one single-element rank-1 ``axis=0`` scatter
    # each -- the exporter's supported shape) and rebuilding only the created
    # rows once after the scan (four gathers plus four selects) removes four row
    # scatters per iteration and four gene tables from the ``while`` carry.  See
    # the ``speciate`` docstring's "Carried state" paragraph.
    def assign_members(carry, slot):
        """Scan 2: threshold membership, materialising a dead slot on demand."""
        assign, dist, keys_ext, seed_table, created_mask = carry
        key_slot = evoxc.gather(keys_ext, slot, axis=0)
        live = evoxc.logical_not(evoxc.isnan(key_slot))
        unassigned = evoxc.isnan(assign)
        any_unassigned = evoxc.reduce_max(unassigned, axes=0)
        create = evoxc.logical_and(evoxc.logical_not(live), any_unassigned)
        counter = evoxc.gather(keys_ext, species_size, axis=0)
        # ``fetch_first`` reports ``I_INF`` when nothing is unassigned; the
        # gather below is unconditional, so the sentinel is clamped to row ``0``.
        seed = evoxc.reshape(fetch_first(enp.expand_dims(unassigned, 0)), ())
        seed = evoxc.select(evoxc.equal(seed, iinf), zero_i32, seed)
        # Centre rows of this slot, selected before the ``cond`` so the branch
        # body exists once: ``pop[seed]`` for a creating slot, the (closed-over)
        # scan-1 centre row otherwise.  A creating slot is the only one whose
        # row the post-pass has to restore, hence the two carries below.
        centre = selected_centre(create, seed, slot, centre_after1)
        key_slot = evoxc.select(create, counter, key_slot)
        keys_ext = evoxc.scatter(keys_ext, slot, key_slot)
        keys_ext = evoxc.scatter(
            keys_ext,
            species_size,
            evoxc.select(create, evoxc.add(counter, 1.0), counter),
        )
        seed_table = evoxc.scatter(seed_table, slot, seed)
        created_mask = evoxc.scatter(
            created_mask, slot, evoxc.cast(create, evoxc.float32)
        )

        def compute_row(
            centre_node_keys,
            centre_node_attrs,
            centre_conn_keys,
            centre_conn_attrs,
            pop_node_keys,
            pop_node_attrs,
            pop_conn_keys,
            pop_conn_attrs,
        ):
            """Distance row of this slot's centre, ``(P,)`` floats."""
            return distance_fn(
                state,
                single_payload(
                    (
                        centre_node_keys,
                        centre_node_attrs,
                        centre_conn_keys,
                        centre_conn_attrs,
                    )
                ),
                payload_like(
                    population,
                    pop_node_keys,
                    pop_node_attrs,
                    pop_conn_keys,
                    pop_conn_attrs,
                ),
            )

        active = evoxc.logical_or(live, create)
        operands = centre + population_fields
        if gate:
            distances = gated_row(active, compute_row, operands)
        else:
            distances = compute_row(*operands)
        closer = evoxc.logical_or(unassigned, evoxc.less(distances, dist))
        within = evoxc.less(distances, threshold)
        take = evoxc.logical_and(evoxc.logical_and(active, within), closer)
        assign = evoxc.select(take, key_slot, assign)
        dist = evoxc.select(take, distances, dist)
        return (assign, dist, keys_ext, seed_table, created_mask), create

    # Scan 1's centre for slot ``i`` is the carry row ``i``, and no earlier
    # iteration can rewrite it (every ``scatter`` in ``reserve`` writes index
    # ``slot``, and each slot is visited exactly once), so the pre-scan centre
    # table already holds exactly the centre every iteration reads.  The dense
    # path measures that whole table before the scan and each iteration reads
    # its own row with a ``gather``, which keeps the compatibility kernel out of
    # the reservation ``while`` body.
    #
    # ``distance_fn.matrix`` does the measuring inside its own fixed-length
    # ``scan``, one centre per iteration: that region is the memory scope, so
    # only one centre's ``(pop_size, 2 * max_conns, ...)`` transients are live at
    # a time instead of the arrays of all ``species_size`` of them.  A kernel at
    # the top level would keep every one of those alive until the generation
    # graph returns, which is both a peak-RSS and a build cost.
    rows1 = None if gate else distance_fn.matrix(state, centre_payload)

    carry, _ = evoxc.scan(reserve, carry, iota_i32(species_size))
    assign, dist, keys_ext, *centre_after1 = carry
    centre_after1 = tuple(centre_after1)
    carry, created = evoxc.scan(
        assign_members,
        (
            assign,
            dist,
            keys_ext,
            enp.zeros((species_size,), dtype=evoxc.int32),
            enp.zeros((species_size,), dtype=evoxc.float32),
        ),
        iota_i32(species_size),
    )
    assign, dist, keys_ext, seed_table, created_mask = carry
    del dist
    # Rows created by scan 2: ``pop[seed_table]`` where ``created_mask`` is set,
    # the scan-1 table everywhere else (whose dead rows stay the stale integers
    # they already were -- ``create`` only fires on a dead slot).
    created = evoxc.greater(created_mask, zero_f32)
    node_keys, node_attrs, conn_keys, conn_attrs = (
        evoxc.select(
            enp.expand_dims(created, tuple(range(1, len(before.shape)))),
            evoxc.gather(field, seed_table, axis=0),
            before,
        )
        for field, before in zip(population_fields, centre_after1)
    )

    # Post-pass: leftovers, fresh-slot bookkeeping, member counts.
    keys_after = evoxc.slice(keys_ext, (0,), (species_size,))
    live_after = evoxc.logical_not(evoxc.isnan(keys_after))
    live_count = evoxc.cast(
        evoxc.sum(evoxc.cast(live_after, evoxc.float32), axes=0), evoxc.int32
    )
    last_live = evoxc.maximum(evoxc.subtract(live_count, int_const(1)), int_const(0))
    assign = evoxc.select(
        evoxc.isnan(assign), evoxc.gather(keys_after, last_live, axis=0), assign
    )
    generation = scalar_f32(state.generation)
    members = evoxc.equal(enp.expand_dims(keys_after, 1), enp.expand_dims(assign, 0))
    counts = evoxc.sum(evoxc.cast(members, evoxc.float32), axes=1)
    return (
        keys_after,
        evoxc.select(created, neg_inf(), species_state.best_fitness),
        evoxc.select(created, generation, species_state.last_improved),
        evoxc.select(evoxc.isnan(keys_after), nan(), counts),
        assign,
        node_keys,
        node_attrs,
        conn_keys,
        conn_attrs,
        evoxc.gather(keys_ext, species_size, axis=0),
    )
