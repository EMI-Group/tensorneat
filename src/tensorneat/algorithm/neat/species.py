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
* **A nested ``cond`` inside a scan body is allowed.**
  The generation graph keeps one level of control flow, but a ``cond`` may sit
  inside an ``evoxc.scan`` step body when both branches are statically built and
  its predicate is a traced mask -- the gated distance rows of
  :func:`._kernels.speciate` do exactly that, and the nested form is proven on
  ``iree``-llvm-cpu with numpy as the reference.
* **No ``.at[]``, no ``cumsum``, no ``where=``-reductions.**
  ``evoxc.scatter`` is replacement-only, prefix sums come from a triangular
  ``dot`` (``evoxc.cumsum`` exists but does not export on compiler backends),
  and every masked reduction first materialises the mask with
  ``evoxc.select``. Indexing is *axis replacing* (numpy ``take``), never
  ``take_along_axis``, so every per-row gather goes through the flattened
  ``common.tools.flat_gather``.
* **Gene payloads are split and integer-keyed.**
  A dead species slot is recognised by ``isnan(species_keys)`` only; its rows
  keep stale integers (int32 cannot carry NaN) and are never read.  The centre
  genomes are the four split payload arrays of
  :class:`~tensorneat.genome.utils.GenomeTensors`, so an absent gene row is
  spelled with the sentinel of *its own* array and no bitcast is needed.

Sizes: ``S = species_size``, ``P = pop_size``, ``N = max_nodes``,
``E = max_conns``, ``NA = node_gene.attr_width``, ``CK = conn_gene.key_width``,
``CA = conn_gene.attr_width``.

The tensor kernels live in the private :mod:`._kernels` module (graph primitives,
spawn allocation, block flattening, parent pairing, the two speciation scans);
this module owns :class:`SpeciesState` and the documented
:class:`SpeciesController` surface driving them.
"""

from __future__ import annotations

from dataclasses import dataclass, replace as dataclass_replace
from typing import Any, Callable

import evoxc
import evoxc.numpy as enp

from tensorneat.common import State, StatefulBaseClass
from tensorneat.common.errors import TensorNEATError
from tensorneat.common.tools import rank_elements

from . import _kernels
from ._validation import (
    non_negative_int,
    positive_float,
    positive_int,
    unit_interval,
)
from ._kernels import (
    POP_KEY,
    SPLIT_POP_KEYS,
    const as _const,
    first_centre_table as _first_centre_table,
    nan as _nan,
    neg_inf as _neg_inf,
    scalar_f32 as _scalar_f32,
)

#: Allocation modes accepted by ``species_number_calculate_by``.
SPAWN_MODES = ("rank", "fitness")


@dataclass(frozen=True)
class SpeciesState:
    """Immutable species bookkeeping, stored under ``state.species``.

    All fields are batch-first over the static species axis ``S`` and are
    flattened as ordinary pytree leaves by ``evoxc`` (a frozen dataclass needs no
    registration).  ``float32`` is used for every scalar field, including the
    ones that conceptually hold integers, so that NaN can encode a dead slot and
    so that ``generation`` (itself a 0-d float32) can be written into
    ``last_improved`` without a dtype change inside a scan carry.

    The centre genes are the four split payload arrays of
    :class:`~tensorneat.genome.utils.GenomeTensors` -- integers and floats never
    share an array, because ``evoxc`` has no bitcast.

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
        center_node_keys (Tensor): ``(S, N)`` int32, the key column of the centre
            genome's node genes per species slot (``I_INF`` padded).  Dead slots
            keep stale integers that no kernel reads.
        center_node_attrs (Tensor): ``(S, N, NA)`` float32, the node attribute
            columns of the same centre genomes (NaN padded).
        center_conn_keys (Tensor): ``(S, E, CK)`` int32, the key columns of the
            centre genome's connection genes.
        center_conn_attrs (Tensor): ``(S, E, CA)`` float32, the connection
            attribute columns of the same centre genomes.
        next_species_key (Tensor): 0-d float32, the key handed to the next
            species created by :meth:`SpeciesController.speciate`; ``0`` is
            reserved for the initial species, so this starts at ``1``.
    """

    species_keys: Any
    best_fitness: Any
    last_improved: Any
    member_count: Any
    idx2species: Any
    center_node_keys: Any
    center_node_attrs: Any
    center_conn_keys: Any
    center_conn_attrs: Any
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
        self.pop_size = positive_int(pop_size, "pop_size")
        self.species_size = positive_int(species_size, "species_size")
        self.max_stagnation = non_negative_int(max_stagnation, "max_stagnation")
        self.species_elitism = non_negative_int(species_elitism, "species_elitism")
        self.spawn_number_change_rate = unit_interval(
            spawn_number_change_rate, "spawn_number_change_rate"
        )
        self.genome_elitism = non_negative_int(genome_elitism, "genome_elitism")
        self.survival_threshold = unit_interval(
            survival_threshold, "survival_threshold", allow_zero=False
        )
        self.min_species_size = positive_int(min_species_size, "min_species_size")
        self.compatibility_threshold = positive_float(
            compatibility_threshold, "compatibility_threshold"
        )
        if not callable(species_fitness_func):
            raise TensorNEATError(
                f"species_fitness_func must be callable, got {species_fitness_func!r}"
            )
        self.species_fitness_func = species_fitness_func
        if species_number_calculate_by not in SPAWN_MODES:
            raise TensorNEATError(
                "species_number_calculate_by must be one of "
                f"{SPAWN_MODES}, got {species_number_calculate_by!r}"
            )
        self.species_number_calculate_by = species_number_calculate_by
        if self.species_size * self.min_species_size > self.pop_size:
            raise TensorNEATError(
                "species_size * min_species_size must not exceed pop_size, got "
                f"{self.species_size} * {self.min_species_size} > {self.pop_size}"
            )

    # ------------------------------------------------------------------ setup

    def setup(self, state: State, first: Any) -> State:
        """Register ``state.species`` as a :class:`SpeciesState`.

        Slot ``0`` is the single live species of generation ``0``: its key is
        ``0``, it owns the whole population (``idx2species`` all zeros,
        ``member_count[0] == pop_size``), its centre is the first individual's
        genome and ``best_fitness`` starts at ``-inf`` so any finite fitness
        counts as an improvement.

        Args:
            state (State): Shared state, already carrying the population payload
                and ``generation`` from ``NEAT.setup``.
            first (GenomeTensors): The four split payload arrays of **one**
                individual -- ``node_keys (N,)``, ``node_attrs (N, NA)``,
                ``conn_keys (E, CK)``, ``conn_attrs (E, CA)`` -- i.e. the
                individual used as the initial centre (population slot ``0``),
                sliced out of the initialised batch by the caller.

        Returns:
            State: ``state.register(species=SpeciesState(...))``.

        Raises:
            TensorNEATError: If ``first`` does not carry the four payload fields
                at their documented ranks (a population axis left in place would
                silently build an ``(S, P, ...)`` centre table).

        Note:
            Dead slots are created by writing ``NaN`` into ``species_keys``,
            ``best_fitness``, ``last_improved`` and ``member_count``.  The
            integer centre arrays cannot hold NaN, so their dead rows are
            zero-filled and every reader gates them behind the
            ``isnan(species_keys)`` mask; the float centre arrays are NaN-filled.
        """
        try:
            node_keys = first.node_keys
            node_attrs = first.node_attrs
            conn_keys = first.conn_keys
            conn_attrs = first.conn_attrs
        except AttributeError as error:
            raise TensorNEATError(
                "setup expects the four split payload arrays of one individual "
                "(node_keys, node_attrs, conn_keys, conn_attrs), got "
                f"{type(first).__name__} without them"
            ) from error
        for name, tensor, rank in (
            ("first.node_keys", node_keys, 1),
            ("first.node_attrs", node_attrs, 2),
            ("first.conn_keys", conn_keys, 2),
            ("first.conn_attrs", conn_attrs, 2),
        ):
            if len(tensor.shape) != rank:
                raise TensorNEATError(
                    f"{name} must be rank {rank} (one individual, no population "
                    f"axis), got shape {tuple(tensor.shape)}"
                )

        species_size = self.species_size
        dead = enp.full((species_size,), float("nan"), dtype=evoxc.float32)
        return state.register(
            species=SpeciesState(
                species_keys=evoxc.scatter(dead, 0, _const(0.0)),
                best_fitness=evoxc.scatter(dead, 0, _neg_inf()),
                last_improved=evoxc.scatter(dead, 0, _const(0.0)),
                member_count=evoxc.scatter(dead, 0, _const(float(self.pop_size))),
                idx2species=enp.zeros((self.pop_size,), dtype=evoxc.float32),
                center_node_keys=_first_centre_table(
                    species_size, node_keys, 0, evoxc.int32
                ),
                center_node_attrs=_first_centre_table(
                    species_size, node_attrs, float("nan"), evoxc.float32
                ),
                center_conn_keys=_first_centre_table(
                    species_size, conn_keys, 0, evoxc.int32
                ),
                center_conn_attrs=_first_centre_table(
                    species_size, conn_attrs, float("nan"), evoxc.float32
                ),
                next_species_key=_const(1.0),
            )
        )

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

        ``idx2species`` and ``next_species_key`` are deliberately *not* gathered:
        the first is population-indexed and the second is a scalar, so neither
        has a species slot to sort.

        Args:
            state (State): Current state, carrying ``species``, ``generation``
                and ``randkey``.
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
        species_state = state.species
        species_fitness = self._species_fitness(species_state, fitness)
        species_state, species_fitness = self._stagnation(
            species_state, species_fitness, state.generation
        )
        order = self._sort_desc_legacy(species_fitness)
        species_state = self._gather_species(species_state, order)

        if self.species_number_calculate_by == "rank":  # static Python branch
            spawn_number = self._cal_spawn_numbers_by_rank(species_state)
        else:
            spawn_number = self._cal_spawn_numbers_by_fitness(species_state)

        selection_key, next_randkey = evoxc.random.split(state.randkey)
        winner, loser, elite_mask = self._pair_parents(
            species_state, selection_key, spawn_number, fitness
        )
        return (
            state.update(species=species_state, randkey=next_randkey),
            winner,
            loser,
            elite_mask,
        )

    # ------------------------------------------------------------ speciate

    def speciate(self, state: State, distance_fn: Callable, pop: Any = None) -> State:
        """Reassign the new population to species (two fixed-length scans).

        Args:
            state (State): State carrying the *new* population payload, the
                sorted/killed ``species`` from :meth:`update_species` and
                ``generation``.
            distance_fn (Callable): Population-digest compatibility distance with
                the signature ``distance_fn(state, centre: GenomeTensors,
                pop: GenomeTensors) -> distances``, where ``pop`` must be the
                payload the digest was prepared from.  Every centre it receives is
                **population-shaped** -- ``pop_size`` rows, row ``p`` measured
                against population member ``p`` -- and returns one ``(pop_size,)``
                distance row.  Slicing the centre out of the ``(S, ...)`` fields
                and tiling it to ``pop_size`` rows is this method's job, not the
                genome's; ``NEAT.tell`` builds the callable from
                ``genome.prepare_distance`` (see :class:`._distance.DistanceRows`),
                and ``genome.execute_distance`` no longer plugs in directly
                because it re-sorts the population for every centre.
            pop (Any, optional): The current population payload.  Defaults to the
                payload read from ``state`` -- the ``POP_KEY`` entry, or the four
                split ``SPLIT_POP_KEYS`` arrays; an explicit argument wins so a
                caller that already holds the payload need not register it.

        Returns:
            State: ``state.update(species=...)`` with a fully rewritten
            :class:`SpeciesState`: fresh ``idx2species`` (no NaN left),
            re-elected centres, possibly new species slots, recomputed
            ``member_count`` and the advanced ``next_species_key``.

        Raises:
            TensorNEATError: If neither the explicit ``pop`` argument nor a
                documented ``state`` entry supplies the population payload.

        Scan 1 -- centre reservation:
            For each slot ``i`` in ``enp.arange(S)``:

            * ``d = distance_fn(state, centre_i, pop)``, shape ``(P,)``;
            * ``closest = argmin(select(isnan(idx2species), d, +inf))`` -- the
              nearest genome that no *earlier* slot has claimed yet
              (``argmin_with_mask``, so "nothing left to claim" is a mask, never
              an out-of-range index: it is folded into ``inert`` below);
            * write ``idx2species[closest] = species_keys[i]``,
              ``center_*[i] = pop_*[closest]`` and ``o2c[closest] = 0``.

            A dead slot performs no write at all: the assignment ``closest`` is
            computed, then every ``scatter`` is gated by a single ``live and
            any-pending`` mask.  This is precisely the legacy early-exit
            condition "stop at the first dead slot", expressed on a fixed trip
            count.

        Scan 2 -- member assignment:
            For each slot ``i`` in ``enp.arange(S)``:

            * if the slot is live, assign every genome with
              ``d < compatibility_threshold`` that is either still unassigned or
              closer to this centre than to its current one
              (``d < o2c``), writing ``idx2species`` and ``o2c`` through that
              mask;
            * if the slot is dead, materialise a new species: its key is
              ``next_species_key``, its first member is the first unassigned
              genome (``fetch_first``, whose ``I_INF`` "no candidate" sentinel is
              clamped to row ``0`` because the gather happens unconditionally),
              that genome becomes the new centre with ``o2c = 0``, and the very
              same threshold assignment then runs against the new centre;
            * an iteration in which no genome is unassigned performs no write
              (again: mask first, ``scatter`` once), which reproduces the legacy
              "all assigned -> stop" condition without a data-dependent trip
              count.

        Post-pass (outside both scans):
            leftover genomes with ``isnan(idx2species)`` adopt the last *live*
            species key (reachable only when the species cap was hit); slots
            created by scan 2 get ``best_fitness = -inf`` and ``last_improved =
            generation``; ``member_count`` is recomputed as one masked compare-sum
            over the ``(S, P)`` membership matrix rather than a per-slot loop.

        Note:
            The created-slot mask is the scans' own ``ys`` output (each step
            reports whether it materialised a species), which is exactly
            "``best_fitness`` was NaN before this call and the key is live now";
            it is reused instead of re-deriving the same mask with ``isnan`` on
            two fields.

        Note:
            **Below ``_kernels.GATED_ROWS`` both scans compute their distance row
            lazily, behind one ``evoxc.cond`` per iteration.**  The predicate is
            the mask that lets the iteration write -- ``live and any-pending`` for
            scan 1, ``live or create`` for scan 2 -- so a slot that cannot claim a
            genome (scan 1) or take a member (scan 2) never runs the
            compatibility kernel.  The skipped branch yields zeros, which the
            iteration's own write mask discards, so no reader ever consumes a
            row that was not computed: the substitution is bit-identical.  A
            sparse table (1-3 live slots of 20 at pop 8/100) would pay for the
            rows of slots that cannot write, so the lazy path is kept there.
        Note:
            **Above ``_kernels.GATED_ROWS`` scan 1's whole centre table is
            measured *before* the scan** (``distance_fn.matrix``, one
            population-shaped call per centre row -- run inside that method's own
            fixed-length ``scan``, so each centre's kernel transients are scoped
            to one region) and each iteration reads its own row with a
            ``gather``, while scan 2 measures its selected centre per iteration
            -- ``pop[seed]`` for a creating slot, the post-scan-1 carry row
            otherwise.  Scan 1's centre for slot ``i`` is the carry row ``i`` and
            no earlier iteration can rewrite it (every ``scatter`` writes index
            ``slot`` and each slot is visited once), so the pre-scan table holds
            exactly the centre every iteration reads and the rows are the same
            floats (the compatibility kernel is row-independent).  Once the
            species table is full every slot is live and nothing is skippable,
            so a lazy row's ``if`` region costs more than it saves (see
            ``_kernels.GATED_ROWS``).  The gate is a pure static predicate, so
            exactly one of the two paths reaches the graph.  The dense path
            needs the digest-backed callable (``DistanceRows.matrix``) and raises
            ``TensorNEATError`` for any other callable -- there is no silent
            fallback to the slower shape.
        """
        population = self._population(state, pop)
        values = _kernels.speciate(
            state,
            state.species,
            distance_fn,
            population,
            self.pop_size,
            self.species_size,
            self.compatibility_threshold,
            _kernels.gated(self.species_size, self.pop_size),
        )
        return state.update(species=SpeciesState(*values))

    @staticmethod
    def _population(state: State, pop: Any = None) -> Any:
        """Read the current population payload handed to ``distance_fn``.

        An explicit ``pop`` wins; otherwise the payload is read from ``state``
        (the ``POP_KEY`` entry, or the four ``SPLIT_POP_KEYS`` arrays).

        Args:
            state (State): Shared state.
            pop (Any, optional): Explicit payload; wins when given.

        Returns:
            Any: The four split payload arrays of the current population.

        Raises:
            TensorNEATError: If no documented entry supplies the payload.  There
                is no eager or packed fallback: a missing payload is a
                configuration error, never a silently empty population.
        """
        return _kernels.read_population(state, pop)

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
        members = self._membership(species_state)
        masked = evoxc.select(members, enp.expand_dims(fitness, 0), _neg_inf())
        return self.species_fitness_func(masked, axes=1)

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
        generation_value = _scalar_f32(generation)
        improved = evoxc.greater(species_fitness, species_state.best_fitness)
        best_fitness = evoxc.select(
            improved, species_fitness, species_state.best_fitness
        )
        last_improved = evoxc.select(
            improved, generation_value, species_state.last_improved
        )
        # ``rank_elements`` ranks the last axis of a 2-D table, so the species
        # vector travels through a ``(1, S)`` window and back.
        ranks = evoxc.reshape(
            rank_elements(enp.expand_dims(species_fitness, 0)), (self.species_size,)
        )
        exempt = evoxc.less(ranks, self.species_elitism)
        stale = evoxc.greater(
            evoxc.subtract(generation_value, last_improved),
            float(self.max_stagnation),
        )
        killed = evoxc.logical_and(
            evoxc.logical_and(evoxc.logical_not(improved), stale),
            evoxc.logical_not(exempt),
        )
        return (
            dataclass_replace(
                species_state,
                species_keys=evoxc.select(
                    killed, _nan(), species_state.species_keys
                ),
                best_fitness=evoxc.select(killed, _nan(), best_fitness),
                last_improved=evoxc.select(killed, _nan(), last_improved),
                member_count=evoxc.select(
                    killed, _nan(), species_state.member_count
                ),
            ),
            evoxc.select(killed, _neg_inf(), species_fitness),
        )

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

        ``spawn = scatter(spawn, h, spawn[h] + (pop_size - sum(spawn)))``
        with ``h`` the first *live* slot (slot ``0`` after the sort).

        Args:
            species_state (SpeciesState): Species state *after* the descending
                sort, so slot order equals fitness rank.

        Returns:
            Tensor: ``(S,)`` int32 with ``sum(spawn) == pop_size`` and dead slots
            at ``0``.

        Note:
            The residual is added to the first live slot, which after the
            descending sort is slot ``0`` (the fittest species); gating on the
            liveness mask keeps a dead slot at ``0`` even for a hand-built,
            unsorted table.  ``species_size * min_species_size <= pop_size`` is
            validated upstream, so the corrected block stays positive.

        Note:
            **Defect fix (all-dead / empty table).**  The legacy denominator
            divided by zero when every slot was dead; here it is floored at
            ``1.0``, so an empty species list produces a finite (all ``-inf``,
            i.e. all-zero) target instead of NaN.

        Note:
            **Defect fix (NaN -> int32 cast).**  The legacy implementation relied
            on the platform behaviour of casting a NaN float to int32 to get
            ``0`` for dead slots.  Here dead slots (and any slot whose damped
            value is NaN) are explicitly written to ``0.0`` *before* the cast, so
            the result does not depend on the cast's NaN semantics.
        """
        return _kernels.spawn_numbers_by_rank(
            species_state.species_keys,
            species_state.member_count,
            self.pop_size,
            self.species_size,
            self.min_species_size,
            self.spawn_number_change_rate,
        )

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
            ``0`` instead of turning every rate into NaN.  A freshly created
            species starts at ``best_fitness = -inf``; the ``nan_to_num`` pass
            turns the resulting non-finite weight into ``0.0``, so such a slot is
            carried by the ``min_species_size`` floor alone.
        """
        return _kernels.spawn_numbers_by_fitness(
            species_state.species_keys,
            species_state.best_fitness,
            species_state.member_count,
            self.pop_size,
            self.min_species_size,
            self.spawn_number_change_rate,
        )

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
        ``clamp(floor(survival_threshold * member_count), 1,
        min(member_count, pop_size))`` and the first ``survive_size`` ranks are
        the candidate parents.  The tail of the ranking holds non-members (all
        ``-inf``); it is unreachable because the candidate count never exceeds
        the member count (and the extra ``pop_size`` cap keeps a stale count from
        indexing past the ``(S, P)`` ranking width).

        Args:
            species_state (SpeciesState): Sorted species state.
            selection_key (Any): One rank-0 ``evoxc.random`` key -- the *only*
                random source of parent selection for this generation.
            spawn_number (Tensor): ``(S,)`` int32 from one of the
                :meth:`_cal_spawn_numbers_by_rank` /
                :meth:`_cal_spawn_numbers_by_fitness` helpers, summing to
                ``pop_size``.
            fitness (Tensor): ``(P,)`` float32, sanitised and maximised.

        Returns:
            tuple: ``(winner, loser, elite_mask)``, all ``(P,)`` -- int32 for the
            two parent row indices into the current population, bool for the
            elite flag.

        Algorithm:
            1. ``slots = evoxc.random.randint(selection_key, (S, 2, P), 0,
               survive[:, None, None])`` -- one draw for both parents: ``high``
               broadcasts over the species axis, the parent axis and the
               population axis, and is ``>= 1`` everywhere.
            2. ``parent = flat_gather(ranking, slots)`` yields ``(S, 2, P)``
               population indices.  ``evoxc.gather`` replaces its axis instead
               of indexing per row, so the per-row draw *must* go through the
               flattened ``common.tools.flat_gather``; the father/mother slices
               then come out of one 0-d ``evoxc.gather`` each.
            3. Winner/loser: the two drawn parents are compared through
               ``gather(fitness, ...)``; ties go to the father (``>=``), matching
               the legacy comparison.  The decision is taken on the *species
               space* draws, before elite pinning, so it is unaffected by which
               population slot ends up owning the pair.
            4. Elite pinning: for ``idx_in_block < genome_elitism`` both parents
               are replaced by the owning species' ``idx_in_block``-th ranked
               member, clamped at ``member_count - 1`` -- the legacy elementwise
               pin, so the block's top ``genome_elitism`` *distinct* members
               survive unmutated (rather than ``genome_elitism`` copies of the
               single best member); the crossover of an elite slot is therefore
               the identity.
            5. Block flattening: ``loc`` and ``idx_in_block`` from
               :meth:`_block_layout` give one gather index
               ``flat = loc * pop_size + idx_in_block`` into the row-major
               ``(S, P)`` parent arrays; a single ``gather`` per array maps the
               per-species draws onto population slots.

        Note:
            **Defect fix (zero-survivor division).**  The legacy code built a
            probability vector ``(p_idx < survive_size) / survive_size``, which
            divides by zero for a one-member species (``floor(threshold * 1) ==
            0``) and fed NaN into ``random.choice``.  Here the survivor count is
            floored at ``1`` *before* it becomes the ``randint`` bound, so such a
            species simply draws rank ``0`` twice -- two identical parents, and
            self-crossover is the identity.
        """
        return _kernels.pair_parents(
            species_state.species_keys,
            species_state.idx2species,
            species_state.member_count,
            selection_key,
            spawn_number,
            fitness,
            self.pop_size,
            self.species_size,
            self.survival_threshold,
            self.genome_elitism,
        )

    # ------------------------------------------------------ block flattening

    def _block_layout(self, spawn_number: Any) -> tuple[Any, Any]:
        """Map each population slot to its species block (the spawn algebra).

        With the exclusive prefix sum ``E`` of ``spawn_number``
        (``E[i] = sum_{j < i} spawn[j]``) and ``K = sum(spawn) == pop_size``,
        population slot ``p`` belongs to

        ``loc(p)      = the last i with spawn[i] > 0 and E[i] <= p``
        ``idx_in_block(p) = p - E[loc(p)]``

        Both are computed without a host loop: the ``(S, P)`` mask
        ``spawn[i] > 0 and E[i] <= p`` is a broadcast comparison of the sparse
        block starts against the slot vector, ``loc`` is the index of the last
        ``True`` in each column (``argmax`` over the mask reversed along the
        species axis), and ``E[loc(p)]`` is one ``gather``.

        Args:
            spawn_number (Tensor): ``(S,)`` int32, summing to ``pop_size``;
                non-positive entries denote absent blocks.

        Returns:
            tuple: ``(loc, idx_in_block)``, both ``(P,)`` int32.

        Note:
            Requiring ``spawn > 0`` is what makes a zero-spawn (dead or empty)
            species consume no population slot -- the same behaviour the legacy
            ``argmax(p_idx < cumsum(spawn))`` produced.
            The block index has to be the *last* qualifying block, not the
            *number* of qualifying blocks: after the species sort only dead
            slots are guaranteed to trail, so a live species can legitimately
            receive ``spawn == 0`` in the middle of the table and a count would
            point at the wrong block.  ``argmax`` returns the first maximum, so
            the mask is reversed first and the index mapped back with
            ``S - 1 - argmax``.
            Because ``sum(spawn) == pop_size`` and ``pop_size >= 1``, at least one
            block starts at or before every slot, so ``loc`` stays in
            ``[0, S - 1]``.
            This method is a **test seam**: production reaches the same algebra
            through ``_kernels.block_layout`` (via ``_pair_parents``), so a
            behavioural change belongs in ``_kernels.py`` and this wrapper only
            keeps the spawn algebra testable without a graph.
        """
        return _kernels.block_layout(spawn_number, self.pop_size, self.species_size)

    # ----------------------------------------------------------- membership

    def _membership(self, species_state: SpeciesState) -> Any:
        """``(S, P)`` membership mask ``member[i, p] = idx2species[p] == key[i]``.

        One primitive, reused by species fitness, member counts and parent
        ranking.  ``NaN`` never compares equal, so a dead slot and an unassigned
        genome both match nothing.

        Args:
            species_state (SpeciesState): Current bookkeeping.

        Returns:
            Tensor: ``(S, P)`` bool.
        """
        return _kernels.membership(species_state.species_keys, species_state.idx2species)

    @staticmethod
    def _gather_species(species_state: SpeciesState, order: Any) -> SpeciesState:
        """Re-apply the species sort to every ``(S, ...)`` field.

        ``idx2species`` is population-indexed and ``next_species_key`` is a
        scalar, so neither is gathered.

        Args:
            species_state (SpeciesState): State to sort.
            order (Tensor): ``(S,)`` int64 permutation from
                :meth:`_sort_desc_legacy`.

        Returns:
            SpeciesState: A new table whose slot ``i`` is the old slot
            ``order[i]``.
        """
        return dataclass_replace(
            species_state,
            species_keys=evoxc.gather(species_state.species_keys, order, axis=0),
            best_fitness=evoxc.gather(species_state.best_fitness, order, axis=0),
            last_improved=evoxc.gather(species_state.last_improved, order, axis=0),
            member_count=evoxc.gather(species_state.member_count, order, axis=0),
            center_node_keys=evoxc.gather(
                species_state.center_node_keys, order, axis=0
            ),
            center_node_attrs=evoxc.gather(
                species_state.center_node_attrs, order, axis=0
            ),
            center_conn_keys=evoxc.gather(
                species_state.center_conn_keys, order, axis=0
            ),
            center_conn_attrs=evoxc.gather(
                species_state.center_conn_attrs, order, axis=0
            ),
        )

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
        return _kernels.sort_desc_legacy(x, axis)

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

        Note:
            This method is a **test seam**: production calls
            ``_kernels.prefix_sum_exclusive`` (via
            ``_kernels.spawn_from_target``), so a behavioural change belongs in
            ``_kernels.py``.
        """
        return _kernels.prefix_sum_exclusive(values)

