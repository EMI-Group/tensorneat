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
* the gene payload is the **four split arrays** of
  :class:`tensorneat.genome.utils.GenomeTensors` (``evoxc`` has no bitcast, so
  integer keys and float attributes are never packed together) with a
  genome-owned padding encoding: this module never inspects padding, it asks the
  genome (``node_exists``, ``conn_exists``, ``max_node_key``,
  ``max_conn_marker``) instead;
* host-side Python is limited to ``__init__`` validation and ``show_details``.

Sizes: ``P = pop_size``, ``S = species_size``, ``N = max_nodes``,
``E = max_conns``, ``NA``/``CK``/``CA`` the node attribute, connection key and
connection attribute widths declared by the genome's genes.
"""

from __future__ import annotations

import inspect
import operator
from typing import Any, Callable, Optional

import evoxc
import evoxc.numpy as enp

from tensorneat.common import State
from tensorneat.common.errors import TensorNEATError
from tensorneat.common.tools import to_numpy

from ...genome.utils import GenomeTensors
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
        Every finite entry is passed through bit-identically (the comparison is
        only used as a ``select`` predicate, never as arithmetic).

    Raises:
        TensorNEATError: If ``fitness`` is not 1-D over the population axis.
    """
    shape = tuple(getattr(fitness, "shape", ()))
    if len(shape) != 1 or shape[0] != pop_size:
        raise TensorNEATError(
            "_sanitize_fitness: fitness must have shape "
            f"({pop_size},) -- one value per individual -- but got "
            f"shape {shape} for {type(fitness).__name__}"
        )
    finite = evoxc.logical_not(evoxc.isnan(evoxc.subtract(fitness, fitness)))
    return evoxc.select(
        finite, fitness, enp.full((), float("-inf"), dtype=fitness.dtype)
    )


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
        TensorNEATError: If any argument is outside its documented domain.  The
            species/spawn knobs are validated by :class:`SpeciesController`; this
            constructor validates ``genome``, ``pop_size`` and
            ``pop_batch_size`` (including that the latter divides the former).

    Note:
        Nothing here builds an ``evoxc`` graph: the constructor only stores host
        configuration, so ``NEAT(...)`` is cheap and side-effect free.
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
        if genome is None:
            raise TensorNEATError(
                "NEAT: genome must not be None; pass a BaseGenome instance "
                "carrying the network mathematics and the gene layout"
            )
        for attribute in ("num_inputs", "num_outputs"):
            # Check the *declaration* rather than the value: a genome exposes
            # these as properties, and evaluating one here would run code that
            # may legitimately need a state.
            declared = hasattr(type(genome), attribute) or attribute in getattr(
                genome, "__dict__", {}
            )
            if not declared:
                raise TensorNEATError(
                    f"NEAT: genome {type(genome).__name__} does not expose "
                    f"'{attribute}'; the algorithm forwards the network arity "
                    "from the genome"
                )

        self._pop_size = _positive_int("pop_size", pop_size)
        self._pop_batch_size = None
        if pop_batch_size is not None:
            batch = _positive_int("pop_batch_size", pop_batch_size)
            if self._pop_size % batch != 0:
                raise TensorNEATError(
                    f"NEAT: pop_batch_size={batch} must divide "
                    f"pop_size={self._pop_size}; the chunked kernels reshape the "
                    "population axis exactly into (pop_size // pop_batch_size, "
                    "pop_batch_size)"
                )
            self._pop_batch_size = batch

        self.genome = genome
        self.species = SpeciesController(
            pop_size=self._pop_size,
            species_size=species_size,
            max_stagnation=max_stagnation,
            species_elitism=species_elitism,
            spawn_number_change_rate=spawn_number_change_rate,
            genome_elitism=genome_elitism,
            survival_threshold=survival_threshold,
            min_species_size=min_species_size,
            compatibility_threshold=compatibility_threshold,
            species_fitness_func=species_fitness_func,
            species_number_calculate_by=species_number_calculate_by,
        )
        # Legacy spelling used by callers/scripts; the same controller object.
        self.species_controller = self.species
        # Cached host-side counting executable for ``show_details`` (see
        # :meth:`_gene_count_executable`); ``None`` until the first call.
        self._details_cache: Optional[tuple[Any, Any]] = None

    # ------------------------------------------------------------------ setup

    def setup(self, state: Optional[State] = None) -> State:
        """Register the population, the generation counter and the species state.

        Order of operations (all host-side Python orchestration around traced
        kernels):

        1. register ``pop_size`` as a plain Python int so the batch-first
           ``genome.initialize`` can size axis 0 without guessing;
        2. ``state = self.genome.setup(state)``;
        3. inside one small compiled graph: split the pipeline-created
           ``randkey`` into an *initialisation* key and the key carried into
           generation 0, call ``genome.initialize(cfg, init_key, pop_size)``
           once for the whole population, create the 0-d ``generation`` counter
           and seed the species controller with population slot ``0`` (one
           individual, no population axis) as the single live species' centre;
        4. ``register`` ``pop`` (a :class:`GenomeTensors`, one leading axis of
           length ``P``), ``generation`` (0-d float32 ``0.0``) and ``species``,
           and ``update`` the already-present ``randkey`` with the advanced key.

        Args:
            state (State, optional): State created by ``Pipeline.setup``,
                carrying at least ``randkey``.  ``None`` starts from an empty
                :class:`State` -- spelled this way because a ``State()`` default
                argument would be evaluated while this module is imported.

        Returns:
            State: New state with ``pop_size``, ``pop``, ``generation``,
            ``species`` and the advanced ``randkey``.

        Raises:
            TensorNEATError: If ``state`` lacks ``randkey`` (the pipeline must
                create it) or the genome's initial tensors do not have the
                documented shapes.

        Note:
            ``generation`` is a 0-d **float32**, not an int, because stagnation
            arithmetic compares it against the float32 ``last_improved`` and
            writes it into that field inside a scan carry.
            ``pop_size``/``pop``/``generation``/``species`` are the only keys
            this algorithm registers; ``randkey`` is only advanced.
            The initialization kernel captures a *static* copy of the state
            (:meth:`_static_state`): the real state still holds the concrete
            ``randkey`` tensor, which must not be baked into a trace.
        """
        if state is None:
            state = State()
        if "randkey" not in state.registered_keys():
            raise TensorNEATError(
                "NEAT.setup: state has no 'randkey'; Pipeline.setup must create "
                "it with evoxc.random.key(seed) before calling setup()"
            )

        state = state.register(pop_size=self._pop_size)
        state = self.genome.setup(state)
        cfg = self._static_state(state)
        options = self._build_options(state)
        pop_size = self._pop_size

        def _init_kernel(randkey):
            init_key, next_key = evoxc.random.split(randkey)
            payload = self.genome.initialize(cfg, init_key, pop_size)
            generation = enp.full((), 0.0, dtype=evoxc.float32)
            # One individual without a population axis: the initial species
            # centre.  ``gather`` with a 0-d index drops axis 0.
            first = GenomeTensors(
                *(evoxc.gather(field, 0, axis=0) for field in payload)
            )
            return [payload, next_key, generation, self._seed_species(cfg, first)]

        executable = evoxc.build(
            _init_kernel, evoxc.TensorSpec((), evoxc.int64), **options
        )
        payload, next_key, generation, species_state = evoxc.run(
            executable, state.randkey
        )
        state = state.register(
            pop=payload, generation=generation, species=species_state
        )
        # ``randkey`` already exists, and ``State.register`` never overwrites, so
        # the advanced key goes through ``update``.
        return state.update(randkey=next_key)

    def _seed_species(self, cfg: State, first: GenomeTensors) -> SpeciesState:
        """Seed the species table with ``first`` as the single live centre.

        Args:
            cfg (State): Static (tensor-free) state carrying the genome layout.
            first (GenomeTensors): One individual without a population axis.

        Returns:
            SpeciesState: The generation-0 species table.

        Note:
            ``SpeciesController.setup`` has two documented spellings -- one
            taking the whole payload, one taking the node/connection key rows --
            so the arity is read from the bound method once at trace time
            instead of pinning this module to one of them.
        """
        setup_fn = self.species.setup
        if len(inspect.signature(setup_fn).parameters) == 2:
            return setup_fn(cfg, first).species
        return setup_fn(cfg, first.node_keys, first.conn_keys).species

    def _static_state(self, state: State) -> State:
        """Return a tensor-free copy of ``state`` for capture inside a trace.

        The host-side configuration every kernel needs (gene objects, operator
        objects, static sizes, pipeline options) lives on ``state`` next to the
        runtime tensors.  Only the tensors must stay out of a trace, because
        baking a concrete ``evoxc.Tensor`` into a graph would freeze the current
        generation's data; everything else is genuine static configuration.

        Args:
            state (State): Any state carrying host configuration.

        Returns:
            State: A new state with every ``evoxc.Tensor`` entry dropped and
            ``pop_size`` forced to this algorithm's static population size.
        """
        entries = {}
        for name in state.registered_keys():
            value = getattr(state, name)
            if isinstance(value, evoxc.Tensor):
                continue
            entries[name] = value
        entries["pop_size"] = self._pop_size
        return State(**entries)

    def _build_options(self, state: State) -> dict:
        """Collect the ``evoxc.build`` keyword arguments registered on ``state``.

        ``Pipeline`` forwards its backend selection by registering plain host
        values, so the compile options are read from (never written to) the
        state:

        * ``backend`` -- defaults to ``"numpy"`` (the reference interpreter)
          when the pipeline registers none;
        * ``device`` -- forwarded only when registered and not ``None``;
        * ``compile_options`` -- a mapping expanded into ``evoxc.build``'s
          ``**options`` (each stage picks the keys it understands).

        Args:
            state (State): State as handed to :meth:`setup` / used on the host
                side.

        Returns:
            dict: Keyword arguments for ``evoxc.build``.
        """
        registered = state.registered_keys()
        options = {"backend": state.backend if "backend" in registered else "numpy"}
        if "device" in registered and state.device is not None:
            options["device"] = state.device
        if "compile_options" in registered and state.compile_options:
            options.update(dict(state.compile_options))
        return options

    # ------------------------------------------------------- generation cycle

    def ask(self, state: State) -> GenomeTensors:
        """Return the population to evaluate; costs zero ops.

        Args:
            state (State): Current state.

        Returns:
            GenomeTensors: ``node_keys (P, N) int32``, ``node_attrs (P, N, NA)
            float32``, ``conn_keys (P, E, CK) int32`` and ``conn_attrs (P, E,
            CA) float32`` -- the gene payload of the whole population, handed to
            :meth:`transform` and to the problem.  No key is consumed and the
            state is unchanged.
        """
        return state.pop

    def transform(self, state: State, pop: Any = None) -> Any:
        """Decode the whole population into one batch-first execution schedule.

        Args:
            state (State): Current state.
            pop (Any, optional): Gene payload to decode; ``None`` means
                ``state.pop`` (see :meth:`ask`).

        Returns:
            Any: The genome's ``Transformed`` pytree, individual on axis 0
            (topological level, per-node incoming-edge table of static width
            ``K``, node validity and the input/output slots).
        """
        payload = state.pop if pop is None else pop
        return self.genome.transform(state, payload.node_keys, payload.conn_keys)

    def forward(
        self, state: State, transformed: Any, inputs: Any, pop: Any = None
    ) -> Any:
        """Run the decoded batch of networks.

        Args:
            state (State): Current state.
            transformed (Any): Schedule returned by :meth:`transform`.
            inputs (Any): ``(P, *input_shape)`` activations.
            pop (Any, optional): Gene payload the schedule was derived from;
                ``None`` means ``state.pop``.  The problem layer may pass an
                explicit payload when it evaluates a population other than the
                one stored on the state.

        Returns:
            Any: ``(P, *output_shape)`` activations.
        """
        payload = state.pop if pop is None else pop
        return self.genome.forward(
            state,
            transformed,
            payload.node_keys,
            payload.node_attrs,
            payload.conn_keys,
            payload.conn_attrs,
            inputs,
        )

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
           the updated state plus the parent pairs ``(winner, loser,
           elite_mask)``; it also consumes the *first* of the generation's
           random subkeys.
        4. **Next generation** (:meth:`_create_next_generation`): per-genome
           unique gene keys, crossover, mutation, elitism and the *second*
           split (crossover/mutation/next ``randkey``).
        5. **Re-speciate**: ``SpeciesController.speciate`` rebuilds membership
           against the new population, re-electing centres and possibly creating
           species.
        6. **Commit**: the new ``pop``, the advanced ``randkey``, the incremented
           ``generation`` and the new ``species`` come back on the returned
           state.

        Args:
            state (State): Current state (population that was just evaluated).
            fitness (Any): ``(P,)`` float32 raw fitness, maximised.

        Returns:
            State: State of the next generation.

        Raises:
            TensorNEATError: If ``fitness`` is not ``(pop_size,)``.

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
        fitness = _sanitize_fitness(fitness, self._pop_size)
        generation = evoxc.add(
            evoxc.reshape(evoxc.cast(state.generation, evoxc.float32), ()),
            enp.full((), 1.0, evoxc.float32),
        )
        state = state.update(generation=generation)
        state, winner, loser, elite_mask = self.species.update_species(state, fitness)
        state = self._create_next_generation(state, winner, loser, elite_mask)
        return self.species.speciate(state, self.genome.execute_distance)

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
            State: ``state.update(pop=..., randkey=...)`` with the next
            generation's gene payload and the key carried into the next call.

        Note:
            Crossover is order-sensitive in the legacy contract -- argument order
            is ``(winner, loser)`` -- so the winner/loser decision made by
            ``SpeciesController._pair_parents`` is preserved verbatim.
            Only the mutation is scan-chunked by ``pop_batch_size``; crossover
            runs once over the whole population.
        """
        payload = state.pop
        winners = GenomeTensors(
            *(evoxc.gather(field, winner, axis=0) for field in payload)
        )
        losers = GenomeTensors(
            *(evoxc.gather(field, loser, axis=0) for field in payload)
        )
        cross_key, mut_key, next_randkey = evoxc.random.split_n(state.randkey, 3)
        crossed = self.genome.execute_crossover(state, cross_key, winners, losers)

        new_node_keys, new_conn_markers = self._new_gene_keys(state)

        def _mutate(key: Any, *fields: Any) -> tuple:
            # Reassemble the split payload, mutate it, and report the four
            # resulting leaves so ``_batched`` can treat the call as a
            # leaf-wise kernel like any other.
            genome = GenomeTensors(*fields[:4])
            return tuple(
                self.genome.execute_mutation(
                    state, key, genome, fields[4], fields[5]
                )
            )

        mutated = GenomeTensors(
            *self._batched(
                _mutate, mut_key, *crossed, new_node_keys, new_conn_markers
            )
        )

        next_fields = []
        for crossed_field, mutated_field in zip(crossed, mutated):
            rank = len(crossed_field.shape)
            mask = evoxc.reshape(
                elite_mask, (self._pop_size,) + (1,) * (rank - 1)
            )
            next_fields.append(evoxc.select(mask, crossed_field, mutated_field))

        return state.update(pop=GenomeTensors(*next_fields), randkey=next_randkey)

    def _new_gene_keys(self, state: State) -> tuple[Any, Any]:
        """Allocate per-genome unique keys for the structural mutations.

        A batch-first mutation creates nodes and connections for the whole
        population in one kernel, so every individual needs key ranges that do
        not overlap:

        * ``new_node_keys = max_node_key(state, node_keys) + 1 + enp.arange(P)``
          -- with ``arange(P)`` inserted in the ``"add_node"`` path, each genome
          adds at most one node per mutation call;
        * ``new_conn_markers`` is ``(P, 3)`` int32:
          ``max_conn_marker(state, conn_keys) + 1 + arange(P * 3).reshape(P, 3)``
          when the connection gene carries a ``historical_marker``, otherwise
          ``enp.zeros((P, 3), dtype=evoxc.int32)``.

        Args:
            state (State): State carrying the population.

        Returns:
            tuple: ``(new_node_keys (P,) int32, new_conn_markers (P, 3) int32)``.

        Note:
            The width ``3`` is a genome-level convention: one mutation call may
            add at most three connections per genome.  Both maxima come from the
            genome helpers ``max_node_key``/``max_conn_marker``, which apply the
            padding encoding themselves; this module never infers validity from
            NaN.  The ``+1`` is spelled as an int32 constant tensor because an
            ``evoxc`` addition against a Python int promotes to int64, and the
            gene key columns are int32.
        """
        one = enp.full((), 1, dtype=evoxc.int32)
        node_keys = state.pop.node_keys
        node_base = evoxc.reshape(
            evoxc.cast(self.genome.max_node_key(state, node_keys), evoxc.int32), ()
        )
        new_node_keys = evoxc.add(
            evoxc.add(node_base, one),
            enp.arange(self._pop_size, dtype=evoxc.int32),
        )

        if self.genome.conn_gene.uses_historical_marker:
            conn_base = evoxc.reshape(
                evoxc.cast(
                    self.genome.max_conn_marker(state, state.pop.conn_keys),
                    evoxc.int32,
                ),
                (),
            )
            markers = evoxc.reshape(
                enp.arange(self._pop_size * 3, dtype=evoxc.int32),
                (self._pop_size, 3),
            )
            new_conn_markers = evoxc.add(evoxc.add(conn_base, one), markers)
        else:
            new_conn_markers = enp.zeros(
                (self._pop_size, 3), dtype=evoxc.int32
            )

        return new_node_keys, new_conn_markers

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
        per-chunk results are stacked on the scan's ``y`` output and reshaped
        back to ``(P, ...)``.

        Args:
            kernel (Callable): Batch-first kernel ``kernel(key, *args)``, already
                bound to its state.
            key (Any): Rank-0 key for this kernel, drawn by the caller.
            *args (Any): ``(P, ...)`` tensors, all sharing the population axis 0.

        Returns:
            Any: The kernel's output with the population axis restored: a tuple
            of leaves when the kernel returns a sequence (as
            ``execute_mutation`` does with its ``GenomeTensors``), otherwise the
            single tensor it returned.

        Note:
            The reshape is pure bookkeeping -- element order is preserved -- so
            for a deterministic kernel the chunked and unchunked results are
            bit-identical.  The scan returns the carry unchanged and reports the
            chunk results as its ``y`` output; ``evoxc.scan`` requires at least
            one symbolic ``y`` leaf, so a carry-only loop is impossible.
        """
        batch = self._pop_batch_size
        pop_size = self._pop_size
        if batch is None or batch >= pop_size:
            return kernel(key, *args)

        n_batches = pop_size // batch
        chunked = tuple(
            evoxc.reshape(arg, (n_batches, batch) + tuple(arg.shape[1:]))
            for arg in args
        )
        keys = evoxc.stack(list(evoxc.random.split_n(key, n_batches)))

        # Filled during the (single) trace of the loop body: whether the kernel
        # reports a sequence of leaves or a single tensor, so the chunked result
        # mirrors the unchunked one exactly.
        returns_sequence: list[bool] = []

        def body(carry, step):
            step_key = step[0]
            chunks = step[1:]
            out = kernel(step_key, *chunks)
            is_sequence = isinstance(out, (tuple, list))
            if not returns_sequence:
                returns_sequence.append(is_sequence)
            return carry, list(out) if is_sequence else [out]

        carry = enp.full((), 0.0, dtype=evoxc.float32)
        _, ys = evoxc.scan(body, carry, (keys, *chunked))
        stacked = list(ys) if isinstance(ys, (tuple, list)) else [ys]
        restored = [
            evoxc.reshape(leaf, (pop_size,) + tuple(leaf.shape[2:]))
            for leaf in stacked
        ]
        if returns_sequence and returns_sequence[0]:
            return tuple(restored)
        return restored[0]

    # ---------------------------------------------------------------- host side

    def show_details(self, state: State, fitness: Any) -> None:
        """Print node/connection counts and the live species sizes.

        Args:
            state (State): Current state.
            fitness (Any): ``(P,)`` fitness of the generation just evaluated.

        Returns:
            None: Output goes to stdout, in the legacy layout
            (``\\tnode counts: max: .., min: .., mean: ..`` etc.).

        Note:
            Host-side only: ``evoxc`` exposes no ``io_callback`` and
            ``runtime_call`` exists on the numpy backend only, so this is called
            by the pipeline *after* the compiled step returns.  All tensor reads
            go through ``common.tools.to_numpy``; node/connection validity is
            counted by the genome helpers (``node_exists``/``conn_exists``)
            inside a small graph cached per payload shape, never by re-testing
            the padding encoding here.
        """
        node_counts, conn_counts = self._gene_counts(state, state.pop)
        # NaN marks a dead species slot; ``NaN > 0`` is False, so the filter
        # both drops dead slots and never needs a host-side NaN test.
        member_count = to_numpy(state.species.member_count)
        species_sizes = [int(count) for count in member_count if count > 0]
        print(
            f"\tnode counts: max: {node_counts.max()}, "
            f"min: {node_counts.min()}, mean: {node_counts.mean():.2f}\n",
            f"\tconn counts: max: {conn_counts.max()}, "
            f"min: {conn_counts.min()}, mean: {conn_counts.mean():.2f}\n",
            f"\tspecies: {len(species_sizes)}, {species_sizes}\n",
        )

    def _gene_counts(self, state: State, payload: GenomeTensors) -> tuple[Any, Any]:
        """Count existing node and connection rows per individual, on the host.

        Args:
            state (State): State used to recover the genome's static config.
            payload (GenomeTensors): Population payload to count.

        Returns:
            tuple: ``(node_counts (P,) numpy, conn_counts (P,) numpy)`` int32.
        """
        executable = self._gene_count_executable(state, payload)
        node_counts, conn_counts = evoxc.run(
            executable, payload.node_keys, payload.conn_keys
        )
        return to_numpy(node_counts), to_numpy(conn_counts)

    def _gene_count_executable(self, state: State, payload: GenomeTensors) -> Any:
        """Build (once per payload shape) the graph that counts valid rows.

        Args:
            state (State): State carrying the genome configuration.
            payload (GenomeTensors): Payload whose static shape keys the cache.

        Returns:
            Any: An ``evoxc`` executable mapping
            ``(node_keys, conn_keys)`` to ``(node_counts, conn_counts)``.
        """
        node_keys = payload.node_keys
        conn_keys = payload.conn_keys
        shape_key = (
            tuple(node_keys.shape),
            tuple(conn_keys.shape),
            node_keys.dtype,
            conn_keys.dtype,
        )
        if self._details_cache is not None and self._details_cache[0] == shape_key:
            return self._details_cache[1]

        cfg = self._static_state(state)

        def _count_kernel(node_key_rows, conn_key_rows):
            node_mask = self.genome.node_exists(cfg, node_key_rows)
            conn_mask = self.genome.conn_exists(cfg, conn_key_rows)
            return [
                evoxc.sum(evoxc.cast(node_mask, evoxc.int32), axes=1),
                evoxc.sum(evoxc.cast(conn_mask, evoxc.int32), axes=1),
            ]

        executable = evoxc.build(
            _count_kernel,
            evoxc.TensorSpec(tuple(node_keys.shape), node_keys.dtype),
            evoxc.TensorSpec(tuple(conn_keys.shape), conn_keys.dtype),
            **self._build_options(state),
        )
        self._details_cache = (shape_key, executable)
        return executable

    # ------------------------------------------------------------- properties

    @property
    def num_inputs(self) -> int:
        """Number of network inputs, forwarded from the genome.

        Returns:
            int: Static input arity, checked by ``Pipeline`` against
            ``problem.input_shape[-1]``.
        """
        return self.genome.num_inputs

    @property
    def num_outputs(self) -> int:
        """Number of network outputs, forwarded from the genome.

        Returns:
            int: Static output arity, checked by ``Pipeline`` against
            ``problem.output_shape[-1]``.
        """
        return self.genome.num_outputs

    @property
    def pop_size(self) -> int:
        """Population size, as a static Python int.

        Returns:
            int: Length of axis 0 of every population tensor.  Backed by
            ``self._pop_size`` because this property has no setter (the base
            class declares it read-only and the pipeline reads it).
        """
        return self._pop_size


def _positive_int(name: str, value: Any) -> int:
    """Validate a strictly positive integer configuration value.

    Args:
        name (str): Parameter name used in the error message.
        value (Any): Candidate value; anything ``operator.index`` accepts (a
            Python int or a NumPy integer) is allowed.

    Returns:
        int: ``value`` as a plain Python int.

    Raises:
        TensorNEATError: If ``value`` is a bool, not integral, or not positive.
    """
    if isinstance(value, bool):
        raise TensorNEATError(f"NEAT: {name} must be a positive int, got {value!r}")
    try:
        integer = operator.index(value)
    except TypeError:
        raise TensorNEATError(
            f"NEAT: {name} must be a positive int, got {value!r} "
            f"({type(value).__name__})"
        ) from None
    if integer <= 0:
        raise TensorNEATError(f"NEAT: {name} must be a positive int, got {value!r}")
    return int(integer)
