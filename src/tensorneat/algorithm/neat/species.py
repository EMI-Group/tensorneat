"""Species bookkeeping for NEAT on the EvoX Compiler.

This module owns everything that happens *between* two evaluated generations on the
species axis: per-species fitness, stagnation, the descending species sort, the
spawn-number allocation, parent pairing and the re-speciation of the new
population.

Design constraints that shape every function below:

* **Fixed-length scans, never data-dependent loops.**
  ``evoxc`` de-sugars ``scan`` into a ``while`` loop, so a trip count has to be
  static at codegen time and the loop carry has to keep identical dtypes/shapes.
  Every species loop therefore runs over ``enp.arange(species_size)`` exactly
  ``species_size`` times and masks its writes; the guards that legacy expressed
  with an early-terminating ``while_loop`` become "mask first, ``scatter`` once".
* **No vmap anywhere.**
  ``evoxc`` refuses to vectorize region-bearing control flow, so a per-species
  function would have to be rewritten batch-first anyway. Every helper here is
  written once over a leading species axis ``S = species_size`` and a population
  axis ``P = pop_size``, spelled out explicitly.
* **No ``.at[]``, no ``cumsum``, no ``where=``-reductions.**
  ``evoxc.scatter`` is replacement-only, prefix sums come from a triangular
  ``dot`` (``evoxc.cumsum`` exists but does not export on compiler backends),
  and every masked reduction first materialises the mask with
  ``evoxc.select``.
* **Gene payloads are integers with a genome-owned padding encoding.**
  A dead species slot is recognised by ``isnan(species_keys)`` only; the centre
  gene rows of a dead slot may hold stale integers because they are never read
  (an int32 tensor cannot carry NaN), and no code here inspects gene padding
  directly -- validity always goes through the genome helpers.

Sizes: ``S = species_size``, ``P = pop_size``, ``N = max_nodes``,
``C = max_conns``, ``NL = node_gene.length``, ``CL = conn_gene.length``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import evoxc
import evoxc.numpy as enp

from tensorneat.common import State, StatefulBaseClass
from tensorneat.common.errors import TensorNEATError
from tensorneat.common.tools import argmin_with_mask, fetch_first, rank_elements


@dataclass(frozen=True)
class SpeciesState:
    """Immutable species bookkeeping, stored under ``state.species``.

    All fields are batch-first over the static species axis ``S`` and are
    flattened as ordinary pytree leaves by ``evoxc`` (a frozen dataclass needs no
    registration).  ``float32`` is used for every scalar field, including the
    ones that conceptually hold integers, so that NaN can encode a dead slot and
    so that ``generation`` (itself a 0-d float32) can be written into
    ``last_improved`` without a dtype change inside a scan carry.

    Attributes:
        species_keys (Tensor): ``(S,)`` float32, the unique primary key of each
            species slot.  ``NaN`` marks a dead slot; a dead slot is addressed by
            that mask alone and never by its index.
        best_fitness (Tensor): ``(S,)`` float32, best per-species fitness seen so
            far; ``-inf`` for a freshly created species, ``NaN`` once the species
            has been killed by stagnation.
        last_improved (Tensor): ``(S,)`` float32, the generation value at which
            ``best_fitness`` last increased; ``0`` for the initial species.
        member_count (Tensor): ``(S,)`` float32, number of population members
            currently assigned to the slot; ``NaN`` for a dead slot.
        idx2species (Tensor): ``(P,)`` float32, the species key of each
            individual.  ``NaN`` means "not assigned yet", which only happens
            *inside* :meth:`SpeciesController.speciate`.
        center_nodes (Tensor): ``(S, N, NL)`` int32, the centre genome's node
            gene rows per species slot.  Dead slots keep stale integers that no
            kernel reads.
        center_conns (Tensor): ``(S, C, CL)`` int32, the centre genome's
            connection gene rows per species slot, same dead-slot convention.
        next_species_key (Tensor): 0-d float32, the key handed to the next
            species created by :meth:`SpeciesController.speciate`; ``0`` is
            reserved for the initial species, so this starts at ``1``.
    """

    species_keys: Any
    best_fitness: Any
    last_improved: Any
    member_count: Any
    idx2species: Any
    center_nodes: Any
    center_conns: Any
    next_species_key: Any


class SpeciesController(StatefulBaseClass):
    """Speciation, stagnation, spawn allocation and parent pairing for NEAT.

    The controller is a pure host-side configuration object; every tensor it
    owns lives in :class:`SpeciesState` on the shared ``State``.  Each method
    below runs inside an active ``evoxc`` trace (except :meth:`__init__`), so
    shapes and Python-level configuration are resolved at trace time and nothing
    may branch on a traced value.

    Args:
        pop_size (int): Population size ``P``; static, axis 0 of every population
            tensor.
        species_size (int): Species-slot capacity ``S``; the fixed trip count of
            both speciation scans and the length of every ``(S, ...)`` field.
        max_stagnation (int): Number of consecutive generations a species may
            fail to improve its ``best_fitness`` before it is killed.
        species_elitism (int): Number of best-ranked species exempt from
            stagnation (ranked by species fitness, largest first).
        spawn_number_change_rate (float): Damping factor in ``[0, 1]`` applied to
            the gap between the target and the previous spawn count, so a
            species' size cannot jump in one generation.
        genome_elitism (int): Number of leading slots of every species block that
            are pinned to the species' best member (two identical parents) and
            later skip mutation.
        survival_threshold (float): Fraction of a species' members, best first,
            that are eligible as parents; floored to at least one member.
        min_species_size (int): Lower bound of a live species' spawn count.
        compatibility_threshold (float): Compatibility distance below which a
            genome belongs to a species centre.
        species_fitness_func (Callable, optional): 1-D reduction with the
            signature ``f(x, axes=...)`` used to collapse a species' member
            fitness to one scalar; defaults to ``evoxc.max`` (the legacy
            ``jnp.max`` criterion).  Any replacement must accept ``axes=`` and
            return a rank-0 tensor.
        species_number_calculate_by (str, optional): ``"rank"`` (linear-ranking
            allocation) or ``"fitness"`` (fitness-proportional allocation).
            Resolved by a static Python branch at trace time.

    Raises:
        TensorNEATError: If any argument is out of its documented domain; the
            message names the offending parameter.
    """

    def __init__(
        self,
        pop_size: int,
        species_size: int,
        max_stagnation: int,
        species_elitism: int,
        spawn_number_change_rate: float,
        genome_elitism: int,
        survival_threshold: float,
        min_species_size: int,
        compatibility_threshold: float,
        species_fitness_func: Callable = evoxc.max,
        species_number_calculate_by: str = "rank",
    ):
        raise NotImplementedError

    # ------------------------------------------------------------------ setup

    def setup(self, state: State, first_nodes: Any, first_conns: Any) -> State:
        """Register ``state.species`` as a :class:`SpeciesState`.

        Slot ``0`` is the single live species of generation ``0``: its key is
        ``0``, it owns the whole population (``idx2species`` all zeros,
        ``member_count[0] == pop_size``), its centre is the first individual's
        genome and ``best_fitness`` starts at ``-inf`` so any finite fitness
        counts as an improvement.

        Args:
            state (State): Shared state, already carrying ``pop_nodes``,
                ``pop_conns`` and ``generation`` from ``NEAT.setup``.
            first_nodes (Tensor): ``(N, NL)`` int32, node gene rows of the
                individual used as the initial centre (population slot ``0``).
            first_conns (Tensor): ``(C, CL)`` int32, connection gene rows of the
                same individual.

        Returns:
            State: ``state.register(species=SpeciesState(...))``.

        Note:
            Dead slots are created by writing ``NaN`` into ``species_keys``,
            ``best_fitness``, ``last_improved`` and ``member_count``.  The centre
            tensors cannot hold NaN (int32 payload), so they are initialised to
            zeros and every reader must gate them behind the
            ``isnan(species_keys)`` mask.
        """
        raise NotImplementedError

    # ------------------------------------------------------- per generation

    def update_species(
        self, state: State, fitness: Any
    ) -> tuple[State, Any, Any, Any]:
        """Turn one generation of fitness into parents for the next generation.

        This is the whole species algebra, in order:

        1. **Per-species fitness** (:meth:`_species_fitness`): the ``(S, P)``
           membership mask ``M[i, p] = idx2species[p] == species_keys[i]`` is
           materialised, the member fitness is selected against ``-inf`` and
           reduced with ``species_fitness_func(..., axes=1)``.  A slot with no
           members therefore scores ``-inf``, never NaN.
        2. **Stagnation** (:meth:`_stagnation`): ``best_fitness`` /
           ``last_improved`` are refreshed on improvement, the top
           ``species_elitism`` ranked species are exempt, and a stagnated species
           is killed by writing NaN through the mask of step 1's key field; its
           species fitness is overwritten with ``-inf`` so the upcoming
           descending sort pushes it to the end.
        3. **Descending species sort** (:meth:`_sort_desc_legacy`): one stable
           ascending ``argsort`` plus an index-reversal gather, applied to every
           ``(S, ...)`` field with a single ``gather`` per field.  After this
           step the slot index *is* the fitness rank, which is what lets the rank
           allocator use ``enp.arange(S)`` directly.
        4. **Spawn allocation**: exactly one of
           :meth:`_cal_spawn_numbers_by_rank` or
           :meth:`_cal_spawn_numbers_by_fitness`, chosen by a static Python
           ``if`` on ``species_number_calculate_by``.  Both return ``(S,)``
           int32 with ``sum(spawn) == pop_size``.
        5. **Parent pairing** (:meth:`_pair_parents`): per-species survivor
           ranking, uniform parent ranks, elite pinning and the block flattening
           that maps population slot ``p`` to ``(loc(p), idx_in_block(p))``.

        Args:
            state (State): Current state, carrying ``species``, ``generation``,
                ``pop_nodes``/``pop_conns`` (needed for member counts only) and
                ``randkey``.
            fitness (Tensor): ``(P,)`` float32, already sanitised by
                ``NEAT.tell`` (NaN and infinities mapped to ``-inf``), maximised.

        Returns:
            tuple: ``(state, winner, loser, elite_mask)`` where

            * ``state`` is updated with the sorted/killed ``species`` and the
              advanced ``randkey``;
            * ``winner`` is ``(P,)`` int32 row indices into the *current*
              population, one per population slot of the next generation;
            * ``loser`` is ``(P,)`` int32, the other parent of each pair;
            * ``elite_mask`` is ``(P,)`` bool, True for slots that keep their
              unmutated crossover child.

        Note:
            **Randomness contract.**  This method performs exactly one
            ``evoxc.random.split`` on ``state.randkey``: the first subkey is the
            only random source of parent pairing, the second is stored back into
            ``state`` for ``NEAT.tell`` to derive the crossover and mutation
            keys from.  Together with the split inside ``NEAT.tell`` this gives
            the two-splits-three-subkeys budget of the algorithm contract, in the
            order parent selection, crossover, mutation.
        """
        raise NotImplementedError

    # ------------------------------------------------------------ speciate

    def speciate(self, state: State, distance_fn: Callable) -> State:
        """Reassign the new population to species (two fixed-length scans).

        Args:
            state (State): State carrying the *new* ``pop_nodes``/``pop_conns``,
                the sorted/killed ``species`` from :meth:`update_species` and
                ``generation``.
            distance_fn (Callable): One-to-population compatibility distance with
                the genome-owned signature
                ``distance_fn(state, center_nodes_i, center_conns_i, pop_nodes,
                pop_conns) -> (P,)``.  Slicing the centre out of the ``(S, ...)``
                fields is this method's job, not the genome's.

        Returns:
            State: ``state.update(species=...)`` with a fully rewritten
            :class:`SpeciesState`: fresh ``idx2species`` (no NaN left),
            re-elected centres, possibly new species slots, recomputed
            ``member_count`` and the advanced ``next_species_key``.

        Scan 1 -- centre reservation:
            For each slot ``i`` in ``enp.arange(S)``:

            * ``d = distance_fn(state, center_nodes[i], center_conns[i],
              pop_nodes, pop_conns)``, shape ``(P,)``;
            * ``closest = argmin(select(isnan(idx2species), d, +inf))`` -- the
              nearest genome that no *earlier* slot has claimed yet;
            * write ``idx2species[closest] = species_keys[i]``,
              ``center_nodes[i] = pop_nodes[closest]``,
              ``center_conns[i] = pop_conns[closest]`` and
              ``o2c[closest] = 0``.

            A dead slot performs no write at all: the assignment ``closest`` is
            computed, then every ``scatter`` is gated by a single
            ``isnan(species_keys[i])`` mask.  This is precisely the legacy
            early-exit condition "stop at the first dead slot", expressed on a
            fixed trip count.

        Scan 2 -- member assignment:
            For each slot ``i`` in ``enp.arange(S)``:

            * if the slot is live, assign every genome with
              ``d < compatibility_threshold`` that is either still unassigned or
              closer to this centre than to its current one
              (``d < o2c``), writing ``idx2species`` and ``o2c`` through that
              mask;
            * if the slot is dead, materialise a new species: its key is
              ``next_species_key``, its first member is the first unassigned
              genome (``fetch_first``, ``-1`` sentinel when there is none), that
              genome becomes the new centre with ``o2c = 0``, and the very same
              threshold assignment then runs against the new centre;
            * an iteration in which no genome is unassigned performs no write
              (again: mask first, ``scatter`` once), which reproduces the legacy
              "all assigned -> stop" condition without a data-dependent trip
              count.

        Post-pass (outside both scans):
            leftover genomes with ``isnan(idx2species)`` adopt the last species
            key (reachable only when the species cap was hit); slots created by
            scan 2 get ``best_fitness = -inf`` and ``last_improved =
            generation``; ``member_count`` is recomputed as one masked compare-sum
            over the ``(S, P)`` membership matrix rather than a per-slot loop.
        """
        raise NotImplementedError

    # ---------------------------------------------------- fitness/stagnation

    def _species_fitness(self, species_state: SpeciesState, fitness: Any) -> Any:
        """Reduce member fitness to one value per species slot.

        Args:
            species_state (SpeciesState): Current bookkeeping.
            fitness (Tensor): ``(P,)`` float32, sanitised and maximised.

        Returns:
            Tensor: ``(S,)`` float32, ``species_fitness_func`` applied along the
            population axis of ``select(membership, fitness, -inf)``.

        Note:
            The ``(S, P)`` membership mask
            ``enp.expand_dims(idx2species, 0) == enp.expand_dims(species_keys, 1)``
            is the single membership primitive of this module; every other
            reduce (member counts, survivor lists) reuses the same expression so
            the NaN comparison semantics are only spelled out once.  Masking with
            ``-inf`` *before* the reduction is what keeps a dead slot at ``-inf``
            instead of letting NaN poison the reduction.
        """
        raise NotImplementedError

    def _stagnation(
        self, species_state: SpeciesState, species_fitness: Any, generation: Any
    ) -> tuple[SpeciesState, Any]:
        """Refresh best fitness / last-improved and kill stagnated species.

        Args:
            species_state (SpeciesState): Current bookkeeping.
            species_fitness (Tensor): ``(S,)`` float32 from
                :meth:`_species_fitness`.
            generation (Tensor): 0-d float32, the generation *that produced*
                ``fitness`` (``NEAT.tell`` increments before calling, matching
                the legacy order).

        Returns:
            tuple: ``(species_state, species_fitness)`` with

            * ``best_fitness``/``last_improved`` updated wherever
              ``species_fitness > best_fitness`` (a NaN comparison is False, so
              dead slots are untouched);
            * ``species_keys``, ``best_fitness``, ``last_improved`` and
              ``member_count`` set to ``NaN`` for a killed species, and its
              ``species_fitness`` set to ``-inf``;
            * centre tensors deliberately left as they are (never read for a dead
              slot and unable to carry NaN).

        Note:
            The stagnation predicate is
            ``(species_fitness <= best_fitness) & (generation - last_improved >
            max_stagnation)``, disabled for the ``species_elitism`` best-ranked
            species (``rank_elements(species_fitness) < species_elitism``).
            Unlike the legacy version there is no ``evoxc.cond`` per species:
            both branches write whole rows, so the whole step is a branch-free
            ``(S,)``-shaped computation of ``select`` calls -- no control flow,
            no scan, nothing to vectorize.
        """
        raise NotImplementedError

    # ---------------------------------------------------- spawn allocation

    def _cal_spawn_numbers_by_rank(self, species_state: SpeciesState) -> Any:
        """Linear-ranking spawn allocation (``species_number_calculate_by="rank"``).

        Linear ranking over the *live* slots of the already sorted state: with
        ``n = sum(~isnan(species_keys))`` live slots, the rank score of slot ``i``
        is ``n - i`` (slot 0 is the fittest because of the sort), the denominator
        is the triangular number ``(n + 1) * n / 2``, and the target is
        ``floor((n - i) / denominator * pop_size)``.

        The target is damped towards the previous size --
        ``previous + (target - previous) * spawn_number_change_rate`` with
        ``previous = member_count`` -- floored at ``min_species_size``, forced to
        ``0`` on dead slots, cast to int32, and finally corrected so the total is
        exactly ``pop_size``:

        ``spawn = scatter(spawn, 0, spawn[0] + (pop_size - sum(spawn)))``

        Args:
            species_state (SpeciesState): Species state *after* the descending
                sort, so slot order equals fitness rank.

        Returns:
            Tensor: ``(S,)`` int32 with ``sum(spawn) == pop_size`` and dead slots
            at ``0``.

        Note:
            The residual is added to slot ``0`` by convention (it is the fittest
            species, and ``species_size * min_species_size <= pop_size`` is
            validated upstream so the corrected block stays positive).

        Note:
            **Defect fix (NaN -> int32 cast).**  The legacy implementation relied
            on the platform behaviour of casting a NaN float to int32 to get
            ``0`` for dead slots.  Here dead slots (and any slot whose damped
            value is NaN) are explicitly written to ``0.0`` *before* the cast, so
            the result does not depend on the cast's NaN semantics.
        """
        raise NotImplementedError

    def _cal_spawn_numbers_by_fitness(self, species_state: SpeciesState) -> Any:
        """Fitness-proportional spawn allocation (``"fitness"`` mode).

        The same damping / clamping / residual-correction path as
        :meth:`_cal_spawn_numbers_by_rank`, but the target rate comes from the
        species' own ``best_fitness``: with ``f' = f - min_live(f) + 1`` (so the
        worst live species still receives a positive weight) the rate is
        ``f' / sum_live(f')`` and the target ``floor(rate * pop_size)``.

        Args:
            species_state (SpeciesState): Species state after the descending
                sort (order irrelevant for this mode).

        Returns:
            Tensor: ``(S,)`` int32 with ``sum(spawn) == pop_size``.

        Note:
            **Defect fix (NaN-poisoned reductions).**  ``min`` and ``sum`` are
            taken over the *live* slots only: the NaN ``best_fitness`` of dead
            slots is masked out with ``select`` before the reduction, so the
            normalization constant is finite and the dead slots stay at spawn
            ``0`` instead of turning every rate into NaN.
        """
        raise NotImplementedError

    # --------------------------------------------------------- parent pairing

    def _pair_parents(
        self,
        species_state: SpeciesState,
        selection_key: Any,
        spawn_number: Any,
        fitness: Any,
    ) -> tuple[Any, Any, Any]:
        """Draw two parents per population slot and decide winner/loser.

        Per species slot, members are ranked by fitness descending
        (``_sort_desc_legacy`` over ``select(membership, fitness, -inf)``,
        applied to the population index vector), the survivor count is
        ``floor(survival_threshold * member_count)`` and the first
        ``max(survive_size, 1)`` ranks are the candidate parents.  The tail of the
        ranking holds non-members (all ``-inf``); it is unreachable because the
        candidate count never exceeds the member count.

        Args:
            species_state (SpeciesState): Sorted species state.
            selection_key (Any): One rank-0 ``evoxc.random`` key -- the *only*
                random source of parent selection for this generation.
            spawn_number (Tensor): ``(S,)`` int32 from one of the ``_cal_spawn_*``
                helpers, summing to ``pop_size``.
            fitness (Tensor): ``(P,)`` float32, sanitised and maximised.

        Returns:
            tuple: ``(winner, loser, elite_mask)``, all ``(P,)`` -- int32 for the
            two parent row indices into the current population, bool for the
            elite flag.

        Algorithm:
            1. ``slots = evoxc.random.randint(selection_key, (S, 2, P), low=0,
               high=enp.expand_dims(max(survive_size, 1), 1))`` -- one draw for
               both parents, shaped so ``high`` broadcasts over the species axis
               and the parent axis.
            2. ``parent = evoxc.gather(sorted_members, slots, axis=1)`` yields
               ``(S, 2, P)`` population indices; the father/mother slices come out
               of a static ``evoxc.slice`` plus ``enp.squeeze``.
            3. Elite pinning: for ``p_idx < genome_elitism`` both parents are
               replaced by ``sorted_members[:, 0]`` (the species' best member), so
               the crossover of an elite slot is the identity.
            4. Block flattening: ``loc`` and ``idx_in_block`` from
               :meth:`_block_layout` give one gather index
               ``flat = loc * pop_size + idx_in_block`` into the row-major
               ``(S, P)`` parent arrays; a single ``gather`` per array maps the
               per-species draws onto population slots.
            5. Winner/loser: ``first``/``second`` are compared through
               ``gather(fitness, ...)``; ties go to ``first``
               (``>=``), matching the legacy comparison.

        Note:
            **Defect fix (zero-survivor division).**  The legacy code built a
            probability vector ``(p_idx < survive_size) / survive_size``, which
            divides by zero for a one-member species (``floor(threshold * 1) ==
            0``) and fed NaN into ``random.choice``.  Here the survivor count is
            floored at ``1`` *before* it becomes the ``randint`` bound, so such a
            species simply draws rank ``0`` twice -- two identical parents, and
            self-crossover is the identity.
        """
        raise NotImplementedError

    # ------------------------------------------------------ block flattening

    def _block_layout(self, spawn_number: Any) -> tuple[Any, Any]:
        """Map each population slot to its species block (the spawn algebra).

        With the exclusive prefix sum ``E`` of ``spawn_number``
        (``E[i] = sum_{j < i} spawn[j]``) and ``K = sum(spawn) == pop_size``,
        population slot ``p`` belongs to

        ``loc(p)      = #{i : spawn[i] > 0 and E[i] <= p} - 1``
        ``idx_in_block(p) = p - E[loc(p)]``

        Both are computed without a host loop: the ``(S, P)`` mask
        ``spawn[i] > 0 and E[i] <= p`` is summed over the species axis by a
        triangular ``dot``-based prefix sum, and ``E[loc(p)]`` is one
        ``gather``.

        Args:
            spawn_number (Tensor): ``(S,)`` int32, summing to ``pop_size``;
                non-positive entries denote absent blocks.

        Returns:
            tuple: ``(loc, idx_in_block)``, both ``(P,)`` int32.

        Note:
            Counting only blocks with ``spawn > 0`` is what makes a zero-spawn
            (dead or empty) species consume no population slot -- the same
            behaviour the legacy ``argmax(p_idx < cumsum(spawn))`` produced, but
            with the division-by-zero-style edge cases removed and with the
            offset available explicitly instead of as a conditional subtraction.
        """
        raise NotImplementedError

    # ------------------------------------------------------------ utilities

    @staticmethod
    def _sort_desc_legacy(x: Any, axis: int = -1) -> Any:
        """Descending sort indices that reproduce the legacy tie order.

        The legacy implementation sorted with a *stable ascending*
        ``argsort`` and reversed the result with ``[::-1]``, so equal values come
        out in reverse index order.  ``evoxc.argsort(..., descending=True)``
        breaks ties in the opposite direction, which would silently change which
        species is "slot 0" and therefore how the spawn residual is distributed.
        The portable reproduction is one stable ascending ``argsort`` followed by
        a gather with the reversal permutation ``n - 1 - enp.arange(n)`` along
        ``axis`` (no ``evoxc.constant`` needed).

        Args:
            x (Tensor): Rank >= 1 tensor whose ``axis`` is sorted.
            axis (int, optional): Axis to sort along; defaults to the last.

        Returns:
            Tensor: int64 indices with ``x``'s shape.
        """
        raise NotImplementedError

    @staticmethod
    def _prefix_sum_exclusive(values: Any) -> Any:
        """Exclusive prefix sum of a ``(S,)`` float vector.

        ``evoxc.cumsum`` exists but does not export on the compiler backends, so
        the prefix sum is a ``dot`` against the static strictly-lower-triangular
        ones matrix: ``dot(tril(ones((S, S)), -1), values[:, None])[:, 0]``.

        Args:
            values (Tensor): ``(S,)`` float32.

        Returns:
            Tensor: ``(S,)`` float32 with ``out[i] = sum_{j < i} values[j]``.
        """
        raise NotImplementedError
