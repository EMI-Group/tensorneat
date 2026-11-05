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

import numpy as np

import evoxc
import evoxc.numpy as enp

from tensorneat.common.errors import TensorNEATError
from tensorneat.common.schedule_accelerator import is_accelerator, kernel_mode
from tensorneat.common.tools import (
    I_INF,
    argmin_with_mask,
    fetch_first,
    flat_gather,
)

from . import _scan_accelerator
from ._validation import distance_centre_mode

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


def scalar_f32(value: Any) -> Any:
    """Normalise a traced value to a rank-0 ``float32``, adding no op when it is.

    A 0-d tensor handed to a graph as an *input* can arrive with shape ``(1,)``,
    and a generation counter may travel in another float dtype, so a scalar that
    is later broadcast against ``(S,)`` fields is normalised here instead of at
    every use site -- but only as far as it needs to be: the ``cast`` is emitted
    only when the static dtype differs and the ``reshape`` only when the static
    rank differs, so a 0-d ``float32`` value (``state.generation``, which
    :meth:`NEAT.setup` creates that way) costs zero dispatches.

    Both branches hold the same value bit for bit: the static check is a Python
    ``if`` over the trace-time dtype/shape, never a predicate on tensor contents.
    """
    if value.dtype != evoxc.float32:
        value = evoxc.cast(value, evoxc.float32)
    if tuple(value.shape) != ():
        value = evoxc.reshape(value, ())
    return value


def payload_like(
    template: Any, node_keys: Any, node_attrs: Any, conn_keys: Any, conn_attrs: Any
) -> Any:
    """Rebuild ``template``'s payload type from the four split arrays.

    A centre row is wrapped in the caller's own payload class
    (``genome.utils.GenomeTensors``) instead of a second, locally defined copy;
    importing ``tensorneat.genome`` from here would make ``algorithm`` depend on
    a sibling node at package import time.
    """
    return type(template)(node_keys, node_attrs, conn_keys, conn_attrs)


def centre_payload(
    population: Any, rows: Any, pop_size: int, tiled: bool
) -> Any:
    """Four 1-D centre rows as the centre payload a distance call takes.

    ``population`` is the payload template and ``pop_size`` the static ``P``;
    ``tiled`` (a static Python choice) selects one rank-raising ``evoxc.broadcast``
    per field against one ``evoxc.reshape`` per field.  Both are one op per field
    and carry the same centre row, so they return the same ``(pop_size,)`` row.
    """
    shape = pop_size if tiled else 1
    fields = (
        evoxc.broadcast(row, (shape,) + tuple(row.shape))
        if tiled
        else evoxc.reshape(row, (shape,) + tuple(row.shape))
        for row in rows
    )
    return payload_like(population, *fields)


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


def block_layout(spawn_number: Any, pop_size: int) -> Any:
    """Map each population slot to its species block: ``(loc, idx_in_block)``.

    With the exclusive prefix sum ``E`` of ``spawn_number`` and
    ``K = sum(spawn) == pop_size``, slot ``p`` belongs to

    ``loc(p)         = the last ``i`` with ``E[i] <= p````
    ``idx_in_block(p) = p - E[loc(p)]``

    The ``spawn[i] > 0`` qualification needs no term of its own under that
    invariant: ``spawn`` is non-negative, so ``E`` is non-decreasing and
    ``E[0] == 0 <= p``; and for the last qualifying block ``i``,
    ``E[i] + spawn[i] == E[i + 1] > p >= E[i]`` forces ``spawn[i] > 0``.  A
    zero-spawn (dead or empty) species therefore still consumes no population
    slot -- its prefix equals the end of the preceding live block, so either an
    earlier live block is the last qualifying one or the invariant is violated.
    The invariant is built by :func:`spawn_from_target` (non-negative entries,
    residual correction making the total exactly ``pop_size``), and
    :func:`pair_parents` is this kernel's only production caller.

    ``loc`` is the *last* qualifying block rather than the number of qualifying
    blocks: zero-spawn species can sit in the middle of the table (after the sort
    only dead slots are guaranteed to trail).  The two forms coincide exactly
    because ``spawn >= 0`` makes ``E`` its running sum, hence monotone
    non-decreasing -- so the indices ``i`` with ``E[i] <= p`` are a *prefix* of
    ``[0, S)``, and "the count of them minus one" IS the last one.  The count is
    spelled ``sum(E[:, None] <= p, axis=0) - 1``: one comparison over the
    ``(S, P)`` mask plus one reduction, with no reversal permutation, no
    ``argmax`` and no ``reverse-index`` constant to bake.  The reduction needs the
    trailing ``cast`` because an integer ``sum`` accumulates in the platform
    integer (``int64``) while ``loc`` is an ``int32`` gather index.
    """
    spawn = evoxc.cast(spawn_number, evoxc.float32)
    exclusive = prefix_sum_exclusive(spawn)
    slots = iota_f32(pop_size)
    mask = evoxc.less_equal(
        enp.expand_dims(exclusive, 1), enp.expand_dims(slots, 0)
    )
    loc = evoxc.cast(
        evoxc.subtract(evoxc.sum(evoxc.cast(mask, evoxc.int32), axes=0), int_const(1)),
        evoxc.int32,
    )
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
    vector), the survivor count is ``clamp(floor(survival_threshold * m), 1,
    max(min(m, pop_size), 1))`` -- ``m`` being the species' ``member_count``
    with a NaN count read as ``0`` -- and the first ``survive_size`` ranks are
    the candidate parents.

    ``survive_size`` is floored at ``1`` *before* it becomes the ``randint``
    bound, which is the fix for the legacy zero-survivor division: a one-member
    species draws rank ``0`` twice, i.e. two identical parents, and
    self-crossover is the identity.  The floor is spelled twice: the ``clamp``
    low bound, and the outer ``max(..., 1)`` on the ceiling, which keeps the
    clamp's two bounds ordered for an empty (``m == 0``) or NaN-count slot so
    such a slot still leaves exactly one candidate parent.  The extra
    ``pop_size`` cap is a no-op for a valid table and keeps the bound inside the
    ``(S, P)`` ranking width.

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
    # One graph constant per distinct scalar, built once and closed over: they are
    # trace-time constants, so a shared binding is one ``constant`` op instead of
    # one per use site.
    minus_inf = neg_inf()
    zero_f32 = const(0.0)
    one_f32 = const(1.0)
    pop_bound = const(float(pop_size))
    pop_i32 = int_const(pop_size)
    zero_i32 = int_const(0)
    one_i32 = int_const(1)
    masked = evoxc.select(members, enp.expand_dims(fitness, 0), minus_inf)
    ranking = evoxc.cast(sort_desc_legacy(masked, axis=1), evoxc.int32)
    previous = evoxc.select(evoxc.isnan(member_count), zero_f32, member_count)
    floored = evoxc.floor(evoxc.multiply(previous, float(survival_threshold)))
    ceiling = evoxc.maximum(evoxc.minimum(previous, pop_bound), one_f32)
    survive = evoxc.cast(evoxc.clamp(floored, one_f32, ceiling), evoxc.int32)
    draws = evoxc.random.randint(
        selection_key,
        (species_size, 2, pop_size),
        0,
        evoxc.reshape(survive, (species_size, 1, 1)),
        dtype=evoxc.int32,
    )
    # ``evoxc.gather`` replaces its axis instead of indexing per row, so the
    # per-row draw must go through the flattened ``common.tools.flat_gather``.
    # ``trusted``: a draw is in ``[0, survive)`` and ``survive`` is clamped to
    # ``[1, pop_size]`` (its ceiling is ``max(min(member_count, pop_size), 1)`` and
    # ``floor(threshold * member_count) <= member_count`` for ``threshold <= 1``),
    # so every index is inside the ranking's row length ``pop_size`` by construction.
    parent = flat_gather(ranking, draws, trusted=True)
    width = species_size * pop_size
    father = evoxc.reshape(evoxc.gather(parent, 0, axis=1), (width,))
    mother = evoxc.reshape(evoxc.gather(parent, 1, axis=1), (width,))
    father_wins = evoxc.greater_equal(
        evoxc.gather(fitness, father, axis=0), evoxc.gather(fitness, mother, axis=0)
    )
    winner_species = evoxc.select(father_wins, father, mother)
    loser_species = evoxc.select(father_wins, mother, father)

    loc, idx_in_block = block_layout(spawn_number, pop_size)
    flat_slot = evoxc.add(evoxc.multiply(loc, pop_i32), idx_in_block)
    winner = evoxc.gather(winner_species, flat_slot, axis=0)
    loser = evoxc.gather(loser_species, flat_slot, axis=0)
    elite_mask = evoxc.less(idx_in_block, genome_elitism)
    # Elite slot ``c`` pins rank ``min(c, live_members - 1)``: the ranking is descending
    # and every finite fitness beats the ``-inf`` a sanitised genome maps to, so the
    # finite members head the row and the clamp stays on a live member.
    alive = evoxc.cast(evoxc.greater(fitness, minus_inf), evoxc.float32)
    live = evoxc.sum(evoxc.select(members, enp.expand_dims(alive, 0), zero_f32), axes=1)
    cap = evoxc.maximum(evoxc.subtract(evoxc.cast(live, evoxc.int32), one_i32), zero_i32)
    ranks = enp.reshape(iota_i32(pop_size), (1, pop_size))
    block_rank = evoxc.minimum(ranks, enp.expand_dims(cap, 1))
    # ``trusted``: ``block_rank = min(rank, cap)`` with ``rank`` in ``[0, P)`` and
    # ``cap`` in ``[0, P - 1]`` (it is a member count minus one), so it never
    # leaves the ranking's row length ``pop_size`` either.
    pinned = evoxc.gather(
        evoxc.reshape(flat_gather(ranking, block_rank, trusted=True), (width,)),
        flat_slot,
        axis=0,
    )
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
#: per-slot ``evoxc.cond`` distance rows.
#:
#: The lazy ``cond`` pays off only while the species table is *sparse*, because a
#: skipped slot still leaves its ``if`` region (and its carry) in the graph: the
#: table is sparse only on small populations (1-3 live slots out of 20 at pop
#: 8/100, 20 of 20 at pop >= 500).  Measured one configuration per fresh process,
#: ``Pipeline.step`` medians of the gated against the ungated tree: 2.3x at pop
#: 100, 1.2x at pop 300, parity from pop 200 up, while an *unconditional*
#: ``cond`` loses 26 % at pop 500 in the nested-``if`` regions alone -- hence the
#: gate, with the threshold between the pop-300 and pop-500 tables.
#: ``GATED_ROWS`` is only compared against static Python ints, so the choice is
#: made at trace time and costs no op.
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
        A pure host-side predicate: the caller branches on it with an ordinary
        Python ``if`` at trace time, never with a traced ``cond``.
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

    **Carried state.**  Both scans carry ``assign``/``dist`` (scan 2 also the
    species-key counter) and the four centre gene tables are rebuilt once after
    each scan: a slot writes only *its own* centre row and ``arange(S)`` visits
    every slot once, so the row another iteration reads is always the pre-scan
    one, and recording ``(source row, did-write)`` per slot and applying it with
    four ``gather`` + four ``select`` afterwards replaces four row scatters per
    iteration.  Scan 1 records ``closest`` (``pop[closest]`` is the re-elected
    centre), scan 2 records ``seed`` (``pop[seed]`` is a creating slot's centre;
    its own centre write was the identity on every other slot).  The host form
    keeps those records in two rank-1 per-slot tables; the accelerator form
    (:mod:`._scan_accelerator`) returns them as the scan's ``y``, so nothing else
    is carried.  A non-writing slot's record is the clamped ``0`` or the
    ``I_INF`` sentinel its mask discards, so the post-scan ``gather`` is always
    in range and never read.

    **Distance rows.**  ``distance_fn`` is the digest-backed callable built by
    ``NEAT.tell`` from ``genome.prepare_distance`` (see
    :class:`._distance.DistanceRows`).  Every call carries one centre spelled by
    ``state.distance_centre`` -- ``"tiled"`` repeats the row to ``pop_size``
    rows, ``"single"`` keeps a leading axis of ``1`` (a host choice on the
    trace-time state, so one graph contains one spelling) -- and returns one
    ``(pop_size,)`` row, so a phase's rows are one call per slot, never one
    ``species_size * pop_size`` batch.

    Which phase form is used is a static choice on ``gate``.  **Sparse table**
    (``gate`` true): rows are computed lazily, one slot at a time, behind a
    single ``evoxc.cond`` per iteration whose predicate is exactly the mask that
    lets that iteration write -- scan 1 ``live and any_pending``, scan 2 ``live
    or create``; a skipped iteration reads a zero row its own write mask
    discards, so the substitution is bit-identical.  **Dense table** (``gate``
    false): scan 1's whole centre table is measured *before* the scan (one call
    per centre row, inside ``matrix``'s fixed-length ``scan``) and each
    iteration gathers its own row, while scan 2 measures its selected centre per
    iteration because a creating slot's centre is data-dependent.  That form
    needs a callable with ``matrix`` (:class:`._distance.DistanceRows`) and
    raises ``TensorNEATError`` for any other, while the gated path still accepts
    an arbitrary callable.

    ``gate`` is a static Python ``bool``, so exactly one of the two paths is
    traced; the distance-centre spelling is a second, independent static choice,
    and both compute the same rows from the same kernel and centre row.
    """
    centre_fields = (
        species_state.center_node_keys,
        species_state.center_node_attrs,
        species_state.center_conn_keys,
        species_state.center_conn_attrs,
    )
    population_fields = tuple(population)
    centre_table = payload_like(population, *centre_fields)
    # Two host choices on the trace-time configuration, each a plain string so
    # one graph contains one route of each pair: the distance call shape
    # (``"tiled"`` hands the genome a population-shaped centre, ``"single"`` a
    # one-individual one) and the formulation mode (:func:`is_accelerator`).
    tiled = distance_centre_mode(state) == "tiled"
    accelerator = is_accelerator(kernel_mode(state))
    # Trace-time constants are built once here and closed over by the scan
    # bodies: a ``while`` body may reference enclosing SSA values, so one
    # ``constant`` op at the top level instead of one re-emitted on each of the
    # body's ``species_size`` iterations.
    iinf = int_const(I_INF)
    inf_f32 = const(float("inf"))
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

    def selected_centre(create, seed, slot):
        """Centre rows of a scan-2 slot: ``pop[seed]`` when creating, else slot's."""
        return tuple(
            evoxc.select(
                create,
                evoxc.gather(field, seed, axis=0),
                evoxc.gather(centre_field, slot, axis=0),
            )
            for field, centre_field in zip(population_fields, centre_after1)
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

    def reserve_compute(
        pop_node_keys,
        pop_node_attrs,
        pop_conn_keys,
        pop_conn_attrs,
        centre_slot,
    ):
        """Distance row of the pre-scan centre ``centre_slot``, ``(P,)`` floats.

        Built *in here* from the four centre tables, so a skipped slot pays
        neither the gather nor the compatibility kernel.  Its centre is row
        ``centre_slot`` of the *pre-scan* tables, which is the row this slot
        would still carry: no earlier iteration writes any row but its own.
        The four population fields are region operands of the lazy ``cond``,
        never host values.
        """
        rows = tuple(
            evoxc.gather(field, centre_slot, axis=0) for field in centre_fields
        )
        return distance_fn(
            state,
            centre_payload(population, rows, pop_size, tiled),
            payload_like(
                population,
                pop_node_keys,
                pop_node_attrs,
                pop_conn_keys,
                pop_conn_attrs,
            ),
        )

    def member_compute(
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
            centre_payload(
                population,
                (
                    centre_node_keys,
                    centre_node_attrs,
                    centre_conn_keys,
                    centre_conn_attrs,
                ),
                pop_size,
                tiled,
            ),
            payload_like(
                population,
                pop_node_keys,
                pop_node_attrs,
                pop_conn_keys,
                pop_conn_attrs,
            ),
        )

    def reserve_row(slot, active):
        """Scan 1's row for ``slot``: one ``cond`` when gated, else the dense table."""
        if gate:
            return gated_row(active, reserve_compute, population_fields + (slot,))
        return evoxc.gather(rows1, slot, axis=0)

    def member_row(centre, active):
        """Scan 2's row for ``centre``: one ``cond`` when gated, else direct."""
        operands = centre + population_fields
        if gate:
            return gated_row(active, member_compute, operands)
        return member_compute(*operands)

    def rebuilt(mask, table, before_fields):
        """Centre tables with row ``i`` set to ``pop[table[i]]`` where ``mask[i]``."""
        return tuple(
            evoxc.select(
                enp.expand_dims(mask, tuple(range(1, len(before.shape)))),
                evoxc.gather(field, table, axis=0),
                before,
            )
            for field, before in zip(population_fields, before_fields)
        )

    def reserve(carry, slot):
        """Scan 1 (host form): claim the nearest unassigned genome for one live slot."""
        assign, dist, keys_ext, closest_table, claim_table = carry
        key_slot = evoxc.gather(keys_ext, slot, axis=0)
        live = evoxc.logical_not(evoxc.isnan(key_slot))
        pending = evoxc.isnan(assign)
        claim = evoxc.logical_and(live, evoxc.reduce_max(pending, axes=0))
        distances = reserve_row(slot, claim)
        # ``argmin_with_mask`` folds "nothing left to claim" into the ``I_INF``
        # sentinel index, which is clamped to row ``0`` before it is used.  Both
        # sentinels are hoisted constants, so a body emits no ``constant`` op.
        closest = evoxc.reshape(
            argmin_with_mask(
                enp.expand_dims(distances, 0), enp.expand_dims(pending, 0),
                filler=inf_f32, default=iinf,
            ),
            (),
        )
        closest = evoxc.select(evoxc.equal(closest, iinf), zero_i32, closest)
        onehot = evoxc.logical_and(evoxc.equal(pop_index, closest), claim)
        assign = evoxc.select(onehot, key_slot, assign)
        dist = evoxc.select(onehot, zero_f32, dist)
        # Record the re-elected centre row instead of scattering it: row ``slot``
        # becomes ``pop[closest]`` where ``claim`` holds, which the post-scan
        # rebuild applies with one gather and one select per field.
        closest_table = evoxc.scatter(closest_table, slot, closest)
        claim_table = evoxc.scatter(claim_table, slot, evoxc.cast(claim, evoxc.float32))
        return (assign, dist, keys_ext, closest_table, claim_table), zero_f32

    def assign_members(carry, slot):
        """Scan 2 (host form): threshold membership, materialising a dead slot."""
        assign, dist, keys_ext, seed_table, created_mask = carry
        key_slot = evoxc.gather(keys_ext, slot, axis=0)
        live = evoxc.logical_not(evoxc.isnan(key_slot))
        unassigned = evoxc.isnan(assign)
        create = evoxc.logical_and(
            evoxc.logical_not(live), evoxc.reduce_max(unassigned, axes=0)
        )
        counter = evoxc.gather(keys_ext, species_size, axis=0)
        # ``fetch_first`` reports ``I_INF`` when nothing is unassigned; the
        # gather below is unconditional, so the sentinel is clamped to row ``0``.
        seed = evoxc.reshape(
            fetch_first(enp.expand_dims(unassigned, 0), default=iinf), ()
        )
        seed = evoxc.select(evoxc.equal(seed, iinf), zero_i32, seed)
        # Centre rows of this slot, selected before the ``cond`` so the branch
        # body exists once: ``pop[seed]`` for a creating slot, the (closed-over)
        # scan-1 centre row otherwise.
        centre = selected_centre(create, seed, slot)
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
        active = evoxc.logical_or(live, create)
        distances = member_row(centre, active)
        closer = evoxc.logical_or(unassigned, evoxc.less(distances, dist))
        within = evoxc.less(distances, threshold)
        take = evoxc.logical_and(evoxc.logical_and(active, within), closer)
        assign = evoxc.select(take, key_slot, assign)
        dist = evoxc.select(take, distances, dist)
        return (assign, dist, keys_ext, seed_table, created_mask), create

    # Scan 1's centre for slot ``i`` is the pre-scan centre row ``i`` and no
    # earlier iteration can rewrite it (every ``scatter`` in ``reserve`` writes
    # index ``slot``, and each slot is visited exactly once), so the dense path
    # measures that whole table before the scan -- inside ``matrix``'s own
    # fixed-length ``scan``, the region that scopes one centre's transients -- and
    # each iteration gathers its own row, keeping the compatibility kernel out of
    # the reservation ``while`` body.
    rows1 = None if gate else distance_fn.matrix(state, centre_table)

    if accelerator:
        # The accelerator formulation (:mod:`._scan_accelerator`) carries no
        # per-slot table: each step returns its record as the scan's ``y``.
        operands = _scan_accelerator.ScanOperands(
            keys=species_state.species_keys,
            pop_index=pop_index,
            iinf=iinf,
            inf_f32=inf_f32,
            zero_i32=zero_i32,
            zero_f32=zero_f32,
            one_f32=const(1.0),
            threshold=threshold,
            gated=gate,
        )
        live_slots = evoxc.logical_not(evoxc.isnan(species_state.species_keys))
        carry, closest_table = evoxc.scan(
            _scan_accelerator.reserve_step(operands, reserve_row),
            (
                enp.full((pop_size,), float("nan"), dtype=evoxc.float32),
                enp.full((pop_size,), float("inf"), dtype=evoxc.float32),
            ),
            iota_i32(species_size),
        )
        assign, dist = carry
        # The raw minimal index is the sentinel exactly when the slot claimed
        # nothing, so the write mask needs no second reduction over ``pending``.
        claimed = evoxc.logical_and(live_slots, evoxc.not_equal(closest_table, iinf))
        centre_after1 = rebuilt(
            claimed, evoxc.select(claimed, closest_table, zero_i32), centre_fields
        )
        carry, seed_table = evoxc.scan(
            _scan_accelerator.assign_step(operands, member_row, selected_centre),
            (assign, dist, species_state.next_species_key),
            iota_i32(species_size),
        )
        assign, _dist, next_key = carry
        created = evoxc.logical_and(
            evoxc.logical_not(live_slots), evoxc.not_equal(seed_table, iinf)
        )
        # The sentinel record of a non-creating slot never addresses a row.
        seed_table = evoxc.select(created, seed_table, zero_i32)
        # Created slot ``i`` took ``next_species_key + (#creates before i)``: the
        # exact sequence the carried counter accumulated in slot order, and exact
        # in ``float32`` because the keys are integral and below ``2**24``.
        keys_after = evoxc.select(
            created,
            evoxc.add(
                species_state.next_species_key,
                prefix_sum_exclusive(evoxc.cast(created, evoxc.float32)),
            ),
            species_state.species_keys,
        )
    else:
        keys_ext = enp.concatenate(
            [
                species_state.species_keys,
                enp.reshape(species_state.next_species_key, (1,)),
            ],
            0,
        )
        carry = (
            enp.full((pop_size,), float("nan"), dtype=evoxc.float32),
            enp.full((pop_size,), float("inf"), dtype=evoxc.float32),
            keys_ext,
            enp.zeros((species_size,), dtype=evoxc.int32),
            enp.zeros((species_size,), dtype=evoxc.float32),
        )
        carry, _ = evoxc.scan(reserve, carry, iota_i32(species_size))
        assign, dist, keys_ext, closest_table, claim_table = carry
        # Scan 1's centre writes, applied once: row ``slot`` becomes
        # ``pop[closest]`` on the slots that claimed, the pre-scan row everywhere
        # else (dead slots keep the stale integers they already held).
        claimed = evoxc.greater(claim_table, zero_f32)
        centre_after1 = rebuilt(claimed, closest_table, centre_fields)
        carry, _ = evoxc.scan(
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
        assign, _dist, keys_ext, seed_table, created_mask = carry
        created = evoxc.greater(created_mask, zero_f32)
        keys_after = evoxc.slice(keys_ext, (0,), (species_size,))
        next_key = evoxc.gather(keys_ext, species_size, axis=0)

    # Rows created by scan 2: ``pop[seed_table]`` where ``created`` is set, the
    # scan-1 table everywhere else (whose dead rows stay the stale integers they
    # already were -- ``create`` only fires on a dead slot).
    node_keys, node_attrs, conn_keys, conn_attrs = rebuilt(
        created, seed_table, centre_after1
    )

    # Post-pass: leftovers, fresh-slot bookkeeping, member counts.
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
        next_key,
    )
