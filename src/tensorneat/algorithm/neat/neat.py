"""NEAT population cycle on the EvoX Compiler.

``NEAT`` is the algorithm the pipeline drives: it owns the population tensors, the
generation counter and the ``SpeciesController`` that shuffles the population
between generations.  A whole generation (``ask`` -> ``transform`` ->
``forward``/evaluate -> ``tell``) is compiled into a single ``evoxc`` graph, so
every method below except :meth:`NEAT.show_details` and :meth:`NEAT.__init__` runs
inside an active trace.

Conventions that hold everywhere in this module:

* population is axis 0 of every population tensor (``P = pop_size``), and the
  batch dimension is written out explicitly -- no ``vmap``;
* control flow appears only at the outermost level of a kernel, and only as
  ``evoxc.scan`` over a static axis (see :meth:`NEAT._batched`);
* gene payloads are integer arrays with a genome-owned padding encoding: this
  module never inspects padding, it asks the genome (``node_exists``,
  ``conn_exists``, ``max_node_key``, ``max_conn_marker``) instead;
* host-side Python is limited to ``__init__`` validation and ``show_details``.

Sizes: ``P = pop_size``, ``S = species_size``, ``N = max_nodes``,
``C = max_conns``, ``NL = node_gene.length``, ``CL = conn_gene.length``.
"""

from __future__ import annotations

from typing import Any, Callable, Optional

import evoxc
import evoxc.numpy as enp

from tensorneat.common import State
from tensorneat.common.errors import TensorNEATError
from tensorneat.common.tools import to_numpy

from ..base import BaseAlgorithm
from .species import SpeciesController, SpeciesState


def _sanitize_fitness(fitness: Any, pop_size: int) -> Any:
    """Map NaN and infinite fitness to ``-inf``.

    Fitness is maximised and a broken topology evaluates to NaN, so the raw
    ``(P,)`` value cannot enter selection.  ``evoxc`` has ``isnan`` but no
    ``isfinite``/``isinf``; the exact composition used here is "``f - f`` is NaN
    for NaN and for both infinities, and ``0`` for every finite value", i.e.

    ``finite = ~isnan(f - f)`` and ``clean = select(finite, f, -inf)``.

    Args:
        fitness (Any): ``(P,)`` float32 raw fitness as produced by the problem.
        pop_size (int): Static population size ``P`` (used to validate the shape
            at trace time).

    Returns:
        Any: ``(P,)`` float32 with every non-finite entry replaced by ``-inf``.

    Raises:
        TensorNEATError: If ``fitness`` is not 1-D over the population axis.
    """
    raise NotImplementedError


class NEAT(BaseAlgorithm):
    """NEAT: tensorised neuroevolution with fitness sharing by speciation.

    Args:
        genome (BaseGenome): The genome definition; supplies the network
            mathematics, the genetic operators and the static gene layout
            (``max_nodes``, ``max_conns``, ``node_gene.length``,
            ``conn_gene.length``, ``num_inputs``, ``num_outputs``).
        pop_size (int): Population size ``P``; axis 0 of every population tensor.
        species_size (int): Maximum number of species slots ``S``.
        max_stagnation (int, optional): Generations a species may fail to improve
            before being killed; defaults to ``15``.
        species_elitism (int, optional): Best-ranked species exempt from
            stagnation; defaults to ``2``.
        spawn_number_change_rate (float, optional): Damping of the spawn-count
            update, in ``[0, 1]``; defaults to ``0.5``.
        genome_elitism (int, optional): Leading slots of each species block that
            are copied unmutated from the species' best member; defaults to
            ``2``.
        survival_threshold (float, optional): Fraction of a species' best members
            eligible as parents, in ``(0, 1]``; defaults to ``0.1``.
        min_species_size (int, optional): Lower bound of a live species' spawn
            count; defaults to ``1``.
        compatibility_threshold (float, optional): Distance below which a genome
            joins a species; defaults to ``2.0``.
        species_fitness_func (Callable, optional): 1-D reduction with signature
            ``f(x, axes=...)`` collapsing a species' member fitness to a scalar;
            defaults to ``evoxc.max``.  It must accept ``axes=`` (not ``axis=``)
            and return a rank-0 tensor -- anything else fails at trace time.
        species_number_calculate_by (str, optional): ``"rank"`` or ``"fitness"``;
            defaults to ``"rank"``.
        pop_batch_size (int, optional): If set, population-level kernels
            (crossover, mutation) are executed as ``evoxc.scan`` chunks of this
            size instead of one full-population call, cutting peak memory from
            ``O(pop_size)`` to ``O(pop_batch_size)``; must divide ``pop_size``.
            Defaults to ``None`` (one full-population call).

    Raises:
        TensorNEATError: If any argument is outside its documented domain.

    Note:
        Chunking with ``pop_batch_size`` changes the random stream: the kernel
        draws population-shaped randomness internally, so a chunk sees
        ``pop_batch_size`` rows of randomness from its own subkey rather than a
        slice of an ``(P, ...)`` draw.  Runs stay reproducible for a fixed
        ``(seed, config, shape)`` -- ``pop_batch_size`` is part of that config.
    """

    def __init__(
        self,
        genome: Any,
        pop_size: int,
        species_size: int = 10,
        max_stagnation: int = 15,
        species_elitism: int = 2,
        spawn_number_change_rate: float = 0.5,
        genome_elitism: int = 2,
        survival_threshold: float = 0.1,
        min_species_size: int = 1,
        compatibility_threshold: float = 2.0,
        species_fitness_func: Callable = evoxc.max,
        species_number_calculate_by: str = "rank",
        pop_batch_size: Optional[int] = None,
    ):
        raise NotImplementedError

    # ------------------------------------------------------------------ setup

    def setup(self, state: State = State()) -> State:
        """Register the population, the generation counter and the species state.

        Order of operations (all host-side Python orchestration around traced
        kernels):

        1. register ``pop_size`` as a plain Python int so the batch-first
           ``genome.initialize`` can size axis 0 without guessing;
        2. ``state = self.genome.setup(state)``;
        3. split the pipeline-created ``randkey`` into an *initialisation* key and
           the key carried into generation 0;
        4. ``pop_nodes, pop_conns = self.genome.initialize(state, init_key)`` --
           one batch-first call covering the whole population;
        5. ``register`` ``pop_nodes`` ``(P, N, NL)``, ``pop_conns``
           ``(P, C, CL)`` and ``generation`` (0-d float32 ``0.0``);
        6. seed the species controller with population slot ``0`` as the single
           live species' centre.

        Args:
            state (State, optional): State created by ``Pipeline.setup``,
                carrying at least ``randkey``.

        Returns:
            State: New state with ``pop_size``, ``pop_nodes``, ``pop_conns``,
            ``generation``, ``species`` and the advanced ``randkey``.

        Raises:
            TensorNEATError: If ``state`` lacks ``randkey`` (the pipeline must
                create it) or the genome's initial tensors do not have the
                documented shapes.

        Note:
            ``generation`` is a 0-d **float32**, not an int, because stagnation
            arithmetic compares it against the float32 ``last_improved`` and
            writes it into that field inside a scan carry.
        """
        raise NotImplementedError

    # ------------------------------------------------------- generation cycle

    def ask(self, state: State) -> tuple[Any, Any]:
        """Return the population to evaluate; costs zero ops.

        Args:
            state (State): Current state.

        Returns:
            tuple: ``(pop_nodes (P, N, NL), pop_conns (P, C, CL))``, the int32
            gene payloads of the whole population.  The tuple *is* the population
            pytree handed to :meth:`transform` and to the problem.
        """
        raise NotImplementedError

    def transform(self, state: State, pop: Any) -> Any:
        """Decode the whole population into one batch-first execution schedule.

        Args:
            state (State): Current state.
            pop (Any): ``(pop_nodes, pop_conns)`` as returned by :meth:`ask`.

        Returns:
            Any: The genome's ``Transformed`` pytree, individual on axis 0
            (topological order, per-node incoming-edge table, depth/level and
            aggregation masks).
        """
        raise NotImplementedError

    def forward(self, state: State, transformed: Any, inputs: Any) -> Any:
        """Run the decoded batch of networks.

        Args:
            state (State): Current state.
            transformed (Any): Schedule returned by :meth:`transform`.
            inputs (Any): ``(P, *input_shape)`` activations.

        Returns:
            Any: ``(P, *output_shape)`` activations.
        """
        raise NotImplementedError

    def tell(self, state: State, fitness: Any) -> State:
        """Consume one generation of fitness and build the next population.

        The algebra, in the order the kernels must run:

        1. **Sanitise fitness**: ``_sanitize_fitness`` maps NaN and both
           infinities to ``-inf``, so an unevaluable topology can never win
           selection.
        2. **Advance the counter**: ``generation += 1`` *before* the species
           update, matching the legacy order -- stagnation is therefore judged
           against the generation that produced the fitness just consumed.
        3. **Species bookkeeping**: ``SpeciesController.update_species`` returns
           the species fitness, stagnation, the descending species sort, the
           spawn allocation and the parent pairs ``(winner, loser, elite_mask)``;
           it also consumes the *first* of the generation's random subkeys.
        4. **New gene keys**: per-genome unique ``new_node_keys`` and
           ``new_conn_markers`` so a batch-first mutation cannot allocate
           colliding keys (see :meth:`_new_gene_keys`).
        5. **Crossover**: gather the winner and loser gene rows with
           ``gather(..., axis=0)`` and run ``genome.execute_crossover`` batch-first
           over ``(P, ...)`` with the second subkey.
        6. **Mutation**: run ``genome.execute_mutation`` batch-first with the third
           subkey, chunked by ``pop_batch_size`` when set.
        7. **Elitism**: ``select(elite_mask[:, None, None], crossover_children,
           mutated_children)`` keeps elite slots unmutated.
        8. **Re-speciate**: ``SpeciesController.speciate`` rebuilds membership
           against the new population, re-electing centres and possibly creating
           species.
        9. **Commit**: store the new ``pop_nodes``/``pop_conns``, the advanced
           ``randkey``, the incremented ``generation`` and the new ``species``.

        Args:
            state (State): Current state (population that was just evaluated).
            fitness (Any): ``(P,)`` float32 raw fitness, maximised.

        Returns:
            State: State of the next generation.

        Note:
            **Randomness budget.**  Exactly two ``evoxc.random`` splits per
            generation:

            * ``selection_key, rest = evoxc.random.split(state.randkey)`` -- the
              first subkey feeds parent selection only (inside
              ``update_species``);
            * ``cross_key, mut_key, next_randkey = evoxc.random.split_n(rest, 3)``
              -- crossover, mutation, and the key carried into
              ``state.randkey``.

            Keys are rank-0 scalars and never travel through a loop carry, so no
            random draw happens inside a control-flow body.
        """
        raise NotImplementedError

    # ------------------------------------------------------------- next pop

    def _create_next_generation(
        self, state: State, winner: Any, loser: Any, elite_mask: Any
    ) -> State:
        """Crossover, mutate and apply elitism for the whole population.

        Args:
            state (State): State after the species update, carrying the current
                population and the advanced ``randkey``.
            winner (Any): ``(P,)`` int32 parent row indices (the fitter parent).
            loser (Any): ``(P,)`` int32 parent row indices (the other parent).
            elite_mask (Any): ``(P,)`` bool, True for slots that must not be
                mutated.

        Returns:
            State: ``state.update(pop_nodes=..., pop_conns=..., randkey=...)``
            with the next generation's gene payloads.

        Note:
            Crossover is order-sensitive in the legacy contract -- argument order
            is ``(winner_nodes, winner_conns, loser_nodes, loser_conns)`` -- so
            the winner/loser decision made by ``SpeciesController._pair_parents``
            is preserved verbatim.
        """
        raise NotImplementedError

    def _new_gene_keys(self, state: State) -> tuple[Any, Any]:
        """Allocate per-genome unique keys for the structural mutations.

        A batch-first mutation creates nodes and connections for the whole
        population in one kernel, so every individual needs key ranges that do
        not overlap:

        * ``new_node_keys = max_node_key(state, pop_nodes) + 1 + enp.arange(P)``
          -- with ``arange(P)`` inserted in the ``"add_node"`` path, each genome
          adds at most one node per mutation call;
        * ``new_conn_markers`` is ``(P, 3)`` int32:
          ``max_conn_marker(state, pop_conns) + 1 + enp.arange(P * 3).reshape(P, 3)``
          when the connection gene carries a ``historical_marker``, otherwise
          ``enp.zeros((P, 3))``.

        Args:
            state (State): State carrying the population.

        Returns:
            tuple: ``(new_node_keys (P,), new_conn_markers (P, 3))``.

        Note:
            The width ``3`` is a genome-level convention: one mutation call may add
            at most three connections per genome.  Both maxima come from the genome
            helpers ``max_node_key``/``max_conn_marker``, which apply the padding
            encoding themselves; this module never infers validity from NaN.
        """
        raise NotImplementedError

    def _batched(self, kernel: Callable, key: Any, *args: Any) -> Any:
        """Run one population kernel, optionally scan-chunked.

        The kernel contract is batch-first: it receives a single rank-0
        ``evoxc.random`` key, draws ``(leading_axis, ...)``-shaped randomness
        internally, and maps ``(P, ...)`` tensors to ``(P, ...)`` tensors.  When
        ``pop_batch_size`` is ``None`` or ``>= pop_size`` the kernel is simply
        called once on the full population.

        Otherwise each argument is reshaped to ``(n_batches, pop_batch_size, ...)``,
        the key is pre-split with ``evoxc.random.split_n(key, n_batches)`` and the
        stacked ``(n_batches,)`` int64 array is passed as the ``xs`` of
        ``evoxc.scan`` -- so the scan step yields a valid **rank-0** key for each
        chunk (``xs`` leaves must be rank >= 1 while keys are rank-0, which is
        exactly why the batch axis and the key axis are separated here).  The
        per-chunk results are stacked along the carry's ``y`` and reshaped back to
        ``(P, ...)``.

        Args:
            kernel (Callable): Batch-first kernel ``kernel(state, key, *args)``.
            key (Any): Rank-0 key for this kernel, drawn by the caller.
            *args (Any): ``(P, ...)`` tensors, all sharing the population axis 0.

        Returns:
            Any: The kernel's output pytree with the population axis restored.

        Note:
            The scan returns the carry unchanged and reports the (unused) chunk
            results as its ``y`` output; ``evoxc.scan`` requires at least one
            symbolic ``y`` leaf, so a carry-only loop is impossible.
        """
        raise NotImplementedError

    # ---------------------------------------------------------------- host side

    def show_details(self, state: State, fitness: Any) -> None:
        """Print node/connection counts and the live species sizes.

        Args:
            state (State): Current state.
            fitness (Any): ``(P,)`` fitness of the generation just evaluated.

        Returns:
            None: Output goes to stdout.

        Note:
            Host-side only: ``evoxc`` exposes no ``io_callback`` and
            ``runtime_call`` exists on the numpy backend only, so this is called
            by the pipeline *after* the compiled step returns.  All tensor reads
            go through ``common.tools.to_numpy``, and node/connection validity is
            counted with the genome helpers rather than by testing the padding
            encoding here.
        """
        raise NotImplementedError

    # ------------------------------------------------------------- properties

    @property
    def num_inputs(self) -> int:
        """Number of network inputs, forwarded from the genome.

        Returns:
            int: Static input arity, checked by ``Pipeline`` against
            ``problem.input_shape[-1]``.
        """
        raise NotImplementedError

    @property
    def num_outputs(self) -> int:
        """Number of network outputs, forwarded from the genome.

        Returns:
            int: Static output arity, checked by ``Pipeline`` against
            ``problem.output_shape[-1]``.
        """
        raise NotImplementedError

    @property
    def pop_size(self) -> int:
        """Population size, as a static Python int.

        Returns:
            int: Length of axis 0 of every population tensor.  Backed by
            ``self._pop_size`` because this property has no setter (the base
            class declares it read-only and the pipeline reads it).
        """
        raise NotImplementedError
