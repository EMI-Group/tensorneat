"""Graph primitives and the guided species kernels behind :mod:`species`.

This private module holds the parts of the species algebra that are written
*against tensors* rather than against the :class:`SpeciesState` container:

* the graph-time constant/index helpers (``enp`` creation ops need a live trace,
  so nothing here may be precomputed on the host);
* the shared building blocks ``sort_desc_legacy``, ``prefix_sum_exclusive`` and
  ``membership``;
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

import evoxc
import evoxc.numpy as enp

from tensorneat.common.errors import TensorNEATError
from tensorneat.common.tools import (
    I_INF,
    argmin_with_mask,
    fetch_first,
    flat_gather,
)

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

    ``distance_fn`` receives a *population-shaped* payload on both sides, so the
    per-slot centre rows are broadcast with ``enp.broadcast_to`` and wrapped in
    the caller's own payload class (``genome.utils.GenomeTensors``) instead of a
    second, locally defined copy -- importing ``tensorneat.genome`` from here
    would make ``algorithm`` depend on a sibling node at package import time.
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

    ``evoxc.cumsum`` exists but does not export on the compiler backends.
    """
    length = values.shape[0]
    triangular = enp.tril(enp.ones((length, length), dtype=evoxc.float32), -1)
    return evoxc.reshape(evoxc.dot(triangular, enp.expand_dims(values, 1)), (length,))


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
    the reversed ``argmax`` (``argmax`` returns the first maximum).
    """
    spawn = evoxc.cast(spawn_number, evoxc.int32)
    exclusive = prefix_sum_exclusive(evoxc.cast(spawn, evoxc.float32))
    slots = iota_f32(pop_size)
    mask = evoxc.logical_and(
        enp.expand_dims(evoxc.greater(spawn, int_const(0)), 1),
        evoxc.less_equal(enp.expand_dims(exclusive, 1), enp.expand_dims(slots, 0)),
    )
    reversed_loc = evoxc.cast(
        evoxc.argmax(evoxc.flip(evoxc.cast(mask, evoxc.int32), 0), axis=0),
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
    does not depend on which population slot ends up owning the pair.  Elite
    slots (``idx_in_block < genome_elitism``) then pin both parents to the owning
    species' best member.
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
    pinned = evoxc.gather(evoxc.gather(ranking, 0, axis=1), loc, axis=0)
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


#: Largest ``species_size * pop_size`` table that is hoisted into one batched
#: ``distance_fn`` call.
#:
#: The compatibility kernel is row-independent, so a ``(k * P)`` batch returns
#: exactly the same floats as ``k`` batches of ``P``; the *cost* is not the same,
#: though.  Per call the numpy interpreter pays a fixed dispatch cost (measured
#: ~2.7 ms) on a hundred-odd ops, while a batch of ``rows`` pairs costs ~5 us per
#: row in population-tiling and buffer allocation.  Batching therefore wins only
#: while the whole table stays small: at ``pop_size=100, species_size=20`` one
#: 2000-row call is ~28 % cheaper than the 20 broadcast calls it replaces, while
#: at ``pop_size=1000`` the same table costs ~10 % more because the interpreter is
#: then row-bound and the 20 separate (in-scan) calls reuse their buffers from a
#: warm cache.  ``TABLE_ROWS`` is the measured crossover: below it the table is
#: hoisted, above it the per-slot call inside the scan is kept.
TABLE_ROWS = 2000


def distance_table(
    state: Any,
    template: Any,
    centres: Any,
    population: Any,
    pop_size: int,
    species_size: int,
    distance_fn: Callable,
) -> Any:
    """``(species_size, pop_size)`` compatibility distances in one batched call.

    ``centres`` and ``population`` are the four split payload arrays.  Every centre
    row is broadcast against the whole population and the resulting
    ``species_size * pop_size`` pairs are handed to ``distance_fn`` as a single
    batch, so the per-call dispatch cost is paid once instead of once per species
    slot.

    Bit-identity: the compatibility kernel is strictly row-independent (elementwise
    per-pair ops plus per-row reductions over the gene axis, one sorted alignment
    per row), so one ``(k * P)`` batch yields exactly the same per-row floats as
    ``k`` batches of ``P``.
    """
    rows = species_size * pop_size
    tiled_centres = []
    for field in centres:
        rest = tuple(field.shape[1:])
        full = enp.broadcast_to(enp.expand_dims(field, 1), (species_size, pop_size) + rest)
        tiled_centres.append(evoxc.reshape(full, (rows,) + rest))
    tiled_population = []
    for field in population:
        rest = tuple(field.shape[1:])
        full = enp.broadcast_to(enp.expand_dims(field, 0), (species_size, pop_size) + rest)
        tiled_population.append(evoxc.reshape(full, (rows,) + rest))
    distances = distance_fn(
        state,
        payload_like(template, *tiled_centres),
        payload_like(template, *tiled_population),
    )
    return evoxc.reshape(distances, (species_size, pop_size))


def speciate(
    state: Any,
    species_state: Any,
    distance_fn: Callable,
    population: Any,
    pop_size: int,
    species_size: int,
    compatibility_threshold: float,
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

    Scan 1's distance rows are hoisted: iteration ``slot`` reads the centre row it
    inherited and only ever writes *its own* row, so the whole
    ``(species_size, pop_size)`` table is the same before the scan starts and is
    computed by :func:`distance_table` (one call when the table fits
    ``TABLE_ROWS``, otherwise the call stays inside the scan, see that constant).
    Scan 2's rows cannot be hoisted as a whole: a *created* species takes the
    first unassigned genome as its centre, so the row a dead slot needs depends on
    the assignments made by every earlier iteration (and the row of *that* genome
    is not in any precomputable table short of a ``pop_size x pop_size`` one).
    Scan 2 therefore keeps the per-slot call.
    """
    keys_ext = enp.concatenate(
        [species_state.species_keys, enp.reshape(species_state.next_species_key, (1,))],
        0,
    )
    carry = (
        enp.full((pop_size,), float("nan"), dtype=evoxc.float32),
        enp.full((pop_size,), float("inf"), dtype=evoxc.float32),
        keys_ext,
        species_state.center_node_keys,
        species_state.center_node_attrs,
        species_state.center_conn_keys,
        species_state.center_conn_attrs,
    )
    # Phase-1 distances are fully precomputable: iteration ``slot`` reads the
    # centre row it inherited and writes only its OWN row, so no earlier iteration
    # can change row ``slot``.  When the whole table fits one interpreter batch the
    # ``species_size`` calls collapse into a single one (bit-identical, see
    # ``distance_table``); above ``TABLE_ROWS`` the per-slot call inside the scan is
    # cheaper and is kept.
    first_distances = (
        distance_table(
            state,
            population,
            (
                species_state.center_node_keys,
                species_state.center_node_attrs,
                species_state.center_conn_keys,
                species_state.center_conn_attrs,
            ),
            tuple(population),
            pop_size,
            species_size,
            distance_fn,
        )
        if species_size * pop_size <= TABLE_ROWS
        else None
    )

    def centre_payload(node_keys, node_attrs, conn_keys, conn_attrs, slot):
        """Broadcast one centre row to a population-shaped payload."""
        return payload_like(
            population,
            enp.broadcast_to(
                evoxc.gather(node_keys, slot, axis=0),
                (pop_size,) + tuple(node_keys.shape[1:]),
            ),
            enp.broadcast_to(
                evoxc.gather(node_attrs, slot, axis=0),
                (pop_size,) + tuple(node_attrs.shape[1:]),
            ),
            enp.broadcast_to(
                evoxc.gather(conn_keys, slot, axis=0),
                (pop_size,) + tuple(conn_keys.shape[1:]),
            ),
            enp.broadcast_to(
                evoxc.gather(conn_attrs, slot, axis=0),
                (pop_size,) + tuple(conn_attrs.shape[1:]),
            ),
        )

    def reserve(carry, slot):
        """Scan 1: claim the nearest unassigned genome for one live slot."""
        assign, dist, keys_ext, node_keys, node_attrs, conn_keys, conn_attrs = carry
        key_slot = evoxc.gather(keys_ext, slot, axis=0)
        live = evoxc.logical_not(evoxc.isnan(key_slot))
        if first_distances is None:
            centre = centre_payload(node_keys, node_attrs, conn_keys, conn_attrs, slot)
            distances = distance_fn(state, centre, population)
        else:
            distances = evoxc.gather(first_distances, slot, axis=0)
        pending = evoxc.isnan(assign)
        any_pending = evoxc.reduce_max(pending, axes=0)
        # ``argmin_with_mask`` folds "nothing left to claim" into the ``I_INF``
        # sentinel index, which is clamped to row ``0`` before it is used.
        closest = evoxc.reshape(
            argmin_with_mask(
                enp.expand_dims(distances, 0), enp.expand_dims(pending, 0)
            ),
            (),
        )
        closest = evoxc.select(
            evoxc.equal(closest, int_const(I_INF)), int_const(0), closest
        )
        claim = evoxc.logical_and(live, any_pending)
        onehot = evoxc.logical_and(evoxc.equal(iota_i32(pop_size), closest), claim)
        assign = evoxc.select(onehot, key_slot, assign)
        dist = evoxc.select(onehot, const(0.0), dist)
        node_keys = evoxc.scatter(
            node_keys,
            slot,
            evoxc.select(
                claim,
                evoxc.gather(population.node_keys, closest, axis=0),
                evoxc.gather(node_keys, slot, axis=0),
            ),
        )
        node_attrs = evoxc.scatter(
            node_attrs,
            slot,
            evoxc.select(
                claim,
                evoxc.gather(population.node_attrs, closest, axis=0),
                evoxc.gather(node_attrs, slot, axis=0),
            ),
        )
        conn_keys = evoxc.scatter(
            conn_keys,
            slot,
            evoxc.select(
                claim,
                evoxc.gather(population.conn_keys, closest, axis=0),
                evoxc.gather(conn_keys, slot, axis=0),
            ),
        )
        conn_attrs = evoxc.scatter(
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

    def assign_members(carry, slot):
        """Scan 2: threshold membership, materialising a dead slot on demand."""
        assign, dist, keys_ext, node_keys, node_attrs, conn_keys, conn_attrs = carry
        key_slot = evoxc.gather(keys_ext, slot, axis=0)
        live = evoxc.logical_not(evoxc.isnan(key_slot))
        unassigned = evoxc.isnan(assign)
        any_unassigned = evoxc.reduce_max(unassigned, axes=0)
        create = evoxc.logical_and(evoxc.logical_not(live), any_unassigned)
        counter = evoxc.gather(keys_ext, species_size, axis=0)
        # ``fetch_first`` reports ``I_INF`` when nothing is unassigned; the
        # gather below is unconditional, so the sentinel is clamped to row ``0``.
        seed = evoxc.reshape(fetch_first(enp.expand_dims(unassigned, 0)), ())
        seed = evoxc.select(evoxc.equal(seed, int_const(I_INF)), int_const(0), seed)
        key_slot = evoxc.select(create, counter, key_slot)
        keys_ext = evoxc.scatter(keys_ext, slot, key_slot)
        keys_ext = evoxc.scatter(
            keys_ext,
            species_size,
            evoxc.select(create, evoxc.add(counter, 1.0), counter),
        )
        node_keys = evoxc.scatter(
            node_keys,
            slot,
            evoxc.select(
                create,
                evoxc.gather(population.node_keys, seed, axis=0),
                evoxc.gather(node_keys, slot, axis=0),
            ),
        )
        node_attrs = evoxc.scatter(
            node_attrs,
            slot,
            evoxc.select(
                create,
                evoxc.gather(population.node_attrs, seed, axis=0),
                evoxc.gather(node_attrs, slot, axis=0),
            ),
        )
        conn_keys = evoxc.scatter(
            conn_keys,
            slot,
            evoxc.select(
                create,
                evoxc.gather(population.conn_keys, seed, axis=0),
                evoxc.gather(conn_keys, slot, axis=0),
            ),
        )
        conn_attrs = evoxc.scatter(
            conn_attrs,
            slot,
            evoxc.select(
                create,
                evoxc.gather(population.conn_attrs, seed, axis=0),
                evoxc.gather(conn_attrs, slot, axis=0),
            ),
        )

        centre = centre_payload(node_keys, node_attrs, conn_keys, conn_attrs, slot)
        distances = distance_fn(state, centre, population)
        active = evoxc.logical_or(live, create)
        closer = evoxc.logical_or(unassigned, evoxc.less(distances, dist))
        within = evoxc.less(distances, float(compatibility_threshold))
        take = evoxc.logical_and(evoxc.logical_and(active, within), closer)
        assign = evoxc.select(take, key_slot, assign)
        dist = evoxc.select(take, distances, dist)
        return (
            assign,
            dist,
            keys_ext,
            node_keys,
            node_attrs,
            conn_keys,
            conn_attrs,
        ), create

    carry, _ = evoxc.scan(reserve, carry, iota_i32(species_size))
    carry, created = evoxc.scan(assign_members, carry, iota_i32(species_size))
    assign, dist, keys_ext, node_keys, node_attrs, conn_keys, conn_attrs = carry
    del dist

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
