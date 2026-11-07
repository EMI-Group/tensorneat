"""``Pipeline`` -- the host-side compile site, generation loop and checkpoint writer.

One *generation* of NEAT is a single ``evoxc`` graph:
``split randkey -> ask -> transform -> problem.evaluate -> tell``.
:meth:`Pipeline.step` is the only place that graph is built, and it is built
**once per shape signature**, never per generation:

* the graph boundary carries only tensors -- the four gene payload arrays, the
  ``generation`` counter, the ``randkey`` and the ``species`` pytree -- while the
  whole tensor-free configuration ``State`` is baked into the graph through the
  closure (:meth:`Pipeline._static_config` drops every tensor and every runtime
  entry), so no concrete ``evoxc.Tensor`` is ever captured by a trace;
* the compiled ``evoxc.Executable`` is cached in :attr:`Pipeline._executables`
  under ``(pop_size, max_nodes, max_conns, max_in_degree, max_depth, key_span)``
  plus the substrate statics and the ``(distance_centre, distance_formulation,
  kernel_mode)`` formulation statics, and reused verbatim while that signature does
  not change, so :meth:`auto_run` pays the compile cost once and then only runs;
* a *different* signature builds a second executable; a population whose shape
  does not match the cached signature is rejected by ``evoxc``'s own shape check,
  which is why nothing here ever reshapes a batch to make it fit.

Randomness discipline: ``evoxc.random`` keys only.  ``step`` performs exactly one
split (``eval_key, tell_key``), hands ``eval_key`` to the problem and
``tell_key`` to ``algorithm.tell`` (which advances it internally), and stores the
resulting key on the returned state -- so exactly one key advance per generation
reaches the caller and ``(seed, config, shape)`` reproduces a run.

The ``ask`` hook is part of the graph: once the state's payload is registered,
the kernel calls ``algorithm.ask(current)`` and hands its result to ``transform``
and to the problem's ``evaluate``, so an algorithm that overrides ``ask`` is
honoured instead of silently bypassed.  Every shipped ``ask`` returns
``state.pop`` unchanged, which makes the call bit-identical to reading the
payload straight off the state.  The host-side report is narrower by contract:
``previous_pop``, the checkpointed genomes and ``tell`` all describe
``state.pop``, so an ``ask`` that returns a *different* population must document
that its evaluated phenotype is not the one reported.

Overflow policy: an execution schedule is lossy when a node has more incoming
edges than the static window ``max_in_degree`` can hold (``edge_overflow > 0``).
The graph returns ``evoxc.max(edge_overflow)``; after the run a positive host
value raises :class:`~tensorneat.common.errors.TensorNEATError` naming the
overflow count, the population size and ``max_in_degree``.  ``max_in_degree`` is
a *genome* static, so the pipeline cannot widen it silently -- the caller
rebuilds the genome -- and a lossy schedule is never used.

Tiled batches: a problem tiles its dataset (``P * S`` rows) or its episodes
(``P * R`` rows) over the population axis, while the genome kernels are
batch-first over ``P``.  The ``act_fn`` this module hands the problem is a
:class:`~tensorneat.host_payload._NetworkActFn`: ``act_fn(inputs)`` is one query
through ``algorithm.prepare_forward``/``algorithm.forward_prepared`` at ``P`` rows
with the tiled inputs as they arrive, and ``act_fn.prepare(batch_rows)`` derives a
pass once for a problem that queries the same batch shape many times (a rollout
loop), so the genome's loop-invariant operands are not re-derived per timestep.
Either way the genome expands its sample-invariant operands itself
(:func:`~tensorneat.genome.window.by_samples`), individual-major as the problem tiles.

A "best genome" is **batch-first**: the selected individual is a one-individual
batch (``node_keys (1, max_nodes)``, ``conn_keys (1, max_conns, key_width)``,
...), i.e. ``state.pop`` at ``P = 1``; only the checkpoint files flatten it.

No ``jax``, no ``torch``, no ``numpy.random``: ``numpy`` appears only in the
host-side bookkeeping of :meth:`Pipeline.auto_run` (logging, checkpointing) and
in :func:`~tensorneat.common.tools.to_numpy`, the one host array accessor.

## Measured cost

No timing is quoted here on purpose: each belongs to a record file that states
its own protocol and names the tree it was measured on.

* ``benchmarks/PERF_NUMPY.md`` -- the hand-maintained numpy per-generation
  record of three measured passes (op count, the tiled func-fit evaluate plus
  RL rollout hoist, and the ``evoxc`` interpreter A/B), each with its own
  before/after pair, protocol, peak RSS and reproduce commands: the current
  numpy cost evidence.
* ``benchmarks/RESULTS.md`` -- the *generated* full-matrix old-vs-new harness
  report, rendered only by ``benchmarks/run_benchmark.py`` from
  ``benchmarks/results/authoritative.json``.
  Its committed record was measured on the merged tree ``63f0fc67`` (3 repeats,
  seeds 42/43/44, 150 generations), so every row describes that tree and that
  session, and only a full harness re-run may replace it.
  Its verdict is ``FAIL``, and the remaining CPU gap on ``xor_p100``,
  ``xor3d_*`` and ``custom_pagie_p500`` is the interpreter overhead those rows
  were measured with, not a kernel cost: the pinned revision (upstream ``main``,
  ``fe85a534``) carries the numpy-interpreter performance pass (per-block
  dispatch plans, a flat value store, dead-value release at last use), and
  ``benchmarks/PERF_NUMPY.md`` pass 3 holds the 1.07-2.25x per-generation A/B
  against the pre-pass revision those rows were measured with.
* ``benchmarks/BACKENDS.md`` -- the measured ``numpy`` vs ``iree`` (llvm-cpu)
  per-generation table, its protocol and the per-gate portability fix record.
* ``benchmarks/results/xor3d_quality_2026-11-04.md`` -- the narrower xor3d and
  Pagie quality record, superseded by the matrix record's quality rows.

Structurally, the numpy reference interpreter pays per traced op *and* per
element while the legacy baseline was the same work fused by XLA, so executed
op count and element volume -- not micro-tuning -- are the levers there.
No CPU compiled backend beats the numpy interpreter on this hardware
(``benchmarks/BACKENDS.md``), so ``"numpy"`` stays the default -- it also needs
no optional dependency -- while ``backend=`` is forwarded verbatim to
``evoxc.build``.  A device is the other case: every measured case is faster on
``device="rocm"`` than on the interpreter, on both compiled paths
(``benchmarks/GPU.md``), so ``device=`` is a real placement request, honoured
end to end by :meth:`Pipeline._place_state` and never a hint a kernel silently
ignores.

The build is one-off and *per shape*: a wider genome
(``max_nodes``/``max_conns``) or a bigger population is a separate compile, which
is why :meth:`Pipeline._executable_for` caches by signature instead of holding one
global executable.
"""

from __future__ import annotations

import datetime
import json
import os
import time
from typing import Any, Callable, Optional

import evoxc
import evoxc.numpy as enp
import numpy as np

from .algorithm import BaseAlgorithm
from .common import State, StatefulBaseClass, TensorNEATError, to_numpy
from .genome.utils import GenomeTensors
from .host_payload import (
    _NetworkActFn,
    _config_value,
    _declared_shape,
    _finite_fitness,
    _formulation_signature,
    _formulation_statics,
    _jsonable,
    _one_individual,
    _schedule_overflow,
    _substrate_signature,
    eval_chunk_rows,
    materialize_payload,
)
from .problem import BaseProblem

__all__ = ["Pipeline"]

#: ``State`` entries holding the *runtime* tensors of one generation; they are
#: rebuilt from the graph's tensor arguments and never baked in as statics.
_RUNTIME_KEYS: tuple[str, ...] = ("pop", "generation", "species", "randkey")

#: ``compile_options`` keys that select *where* a compiled artifact runs. The
#: setup-time graphs run on CPU, so they must not see them; every other option
#: key is forwarded unchanged.
_DEVICE_TARGETING_OPTIONS: tuple[str, ...] = ("device", "target_backends")

#: Backends that own no CPU-capable device, mapped to the backend their
#: **host-only** setup/reporting graphs run on instead. ``"xla"`` drives a PJRT
#: plugin whose platform follows the installed plugin (``rocm`` here), so it
#: refuses to build for ``Device("cpu", 0)``; those graphs are host-side
#: orchestration whose resulting tensors are placed on the real device afterwards
#: (:meth:`Pipeline._place_state`), so the reference interpreter serves them.
#: Every other backend keeps building its own host artifact on llvm-cpu.
_HOST_BACKEND_FALLBACKS: dict[str, str] = {"xla": "numpy"}


# The host payload helpers (``_NetworkActFn``, ``_one_individual``, ...) live in
# :mod:`tensorneat.host_payload` and are imported above.


# --------------------------------------------------------------------------- #
# the pipeline
# --------------------------------------------------------------------------- #


class Pipeline(StatefulBaseClass):
    """Drive one evolutionary algorithm over one problem and log the trajectory.

    Args:
        algorithm (BaseAlgorithm): The algorithm to run (``NEAT`` or
            ``HyperNEAT``); it owns the population tensors and the species
            bookkeeping, and every kernel is routed through it.
        problem (BaseProblem): The task that turns a schedule into fitness.
        seed (int, optional): Seed of the ``evoxc.random.key`` created in
            :meth:`setup` -- the only source of randomness of a run. Defaults to ``42``.
        fitness_target (float, optional): Fitness (maximised) at which
            :meth:`auto_run` stops. Defaults to ``1``.
        generation_limit (int, optional): Maximum number of generations
            :meth:`auto_run` executes. Defaults to ``1000``.
        backend (str, optional): Forwarded verbatim to ``evoxc.build``;
            ``"numpy"`` is the reference interpreter, ``"iree"``/``"tvm"``/
            ``"xla"`` are opt-in extras. Nothing falls back. Defaults to ``"numpy"``.
        device (Any, optional): Forwarded to ``evoxc.build`` only when not
            ``None``; a kind string (``"cpu"``, ``"cuda"``, ``"rocm"``) or an
            ``evoxc.Device``. The setup-time graphs stay on CPU and the runtime
            tensors are placed on this device once, after ``setup``
            (:meth:`setup` / :meth:`_place_state`), so every generation is
            device-resident. Nothing falls back to CPU. Defaults to ``None``.
        compile_options (dict | None, optional): Expanded into ``evoxc.build``'s
            ``**options`` (each stage uses the keys it understands). Defaults to ``None``.
        is_save (bool, optional): Write ``config.txt``, ``log.txt`` and one
            ``genomes/<generation>.npz`` per generation. Defaults to ``False``.
        save_dir (str | os.PathLike | None, optional): Checkpoint directory;
            ``None`` with ``is_save`` creates ``"./Pipeline YYYYmmddHHMMSS"``.
            Defaults to ``None``.
        is_detailed (bool, optional): Print the per-generation report and call
            ``algorithm.show_details``. Defaults to ``True``.
        show_problem_details (bool, optional): Also call the problem's host-side
            ``show_details`` hook each generation. Defaults to ``False``.
        pop_batch_size (int | None, optional): Must agree with the algorithm's own
            ``pop_batch_size``: chunking is fixed when ``NEAT`` is constructed (it
            changes the random stream), so the pipeline cannot inject it.
            ``None`` means "whatever the algorithm was built with". Defaults to ``None``.
        host_payload (bool, optional): Whether :meth:`step` materialises the evaluated
            population as host numpy arrays. ``True`` -- the honest default -- copies
            the four gene arrays to the host every generation (one explicit
            device-to-host transfer per array, ~0.7 ms/generation for a ``P=500``
            payload on the GPU leg) so the returned ``previous_pop`` and everything
            built on it (:attr:`best_genome`, the checkpoints, the problem's
            ``show_details``) is plain numpy. ``False`` skips those copies and hands
            back the payload **as the backend left it** -- numpy on the reference
            interpreter, device-resident tensors on a device backend -- so a caller
            that wants host arrays must materialise them itself with
            ``common.tools.to_numpy``. The reporting paths still work: they copy the
            payload only on the generations they actually need it (a new best, a
            checkpoint, a report hook). Defaults to ``True``.

    Raises:
        TensorNEATError: If the problem and the algorithm disagree about the
            network arity, or if ``pop_batch_size`` disagrees with the
            algorithm's own setting.

    Note:
        Construction is host-only: it validates configuration, stores it and
        prints nothing. No graph is built until the first :meth:`step` (or the
        explicit warm-up inside :meth:`auto_run`).
    """

    def __init__(
        self,
        algorithm: BaseAlgorithm,
        problem: BaseProblem,
        seed: int = 42,
        fitness_target: float = 1,
        generation_limit: int = 1000,
        backend: str = "numpy",
        device: Any = None,
        compile_options: Optional[dict] = None,
        is_save: bool = False,
        save_dir: Optional[Any] = None,
        is_detailed: bool = True,
        show_problem_details: bool = False,
        pop_batch_size: Optional[int] = None,
        host_payload: bool = True,
    ) -> None:
        self.algorithm = algorithm
        self.problem = problem
        self.seed = seed
        self.fitness_target = fitness_target
        self.generation_limit = generation_limit
        self.backend = backend
        self.device = device
        self.compile_options = compile_options
        self.is_save = is_save
        self.is_detailed = is_detailed
        self.show_problem_details = show_problem_details
        self.host_payload = bool(host_payload)

        # Trace-time kernel formulation statics, derived from the device *kind*
        # this run places on (``_formulation_statics``): plain strings registered
        # on the state and baked into the generation graph's closure, never graph
        # inputs.  They change how the kernels are formulated, so they are part of
        # the executable-cache key (``_shape_signature``).
        (
            self.distance_centre,
            self.distance_formulation,
            self.kernel_mode,
        ) = _formulation_statics(self.device)

        self.pop_size = int(algorithm.pop_size)
        self.pop_batch_size = self._resolve_pop_batch_size(pop_batch_size)
        self._validate_arity()

        # In-memory trajectory / best-so-far; reset again by ``setup``.
        # ``best_genome`` is a one-individual batch -- ``state.pop`` at ``P = 1``,
        # so ``(1, max_nodes)`` keys and ``(1, max_nodes, A)`` node attrs.
        self.fitness_history: list[tuple[int, float]] = []
        self.best_fitness: float = float("-inf")
        self.best_genome: Optional[GenomeTensors] = None
        #: First failure per diagnostics hook, keyed by hook label: the
        #: reporting-only record that keeps a broken printer visible.
        self.detail_failures: dict[str, tuple[str, str]] = {}
        #: Host-side generation number of the last :meth:`step`, read out of the same
        #: packed host block as the fitness vector so the report never pays a second
        #: device read for a scalar.
        self._host_generation: Optional[int] = None

        # Per-shape executable cache: signature -> evoxc.Executable.
        self._executables: dict[tuple[Any, ...], Any] = {}
        self.shape_signature: Optional[tuple[Any, ...]] = None
        self.build_seconds: float = 0.0
        self.builds: int = 0

        if is_save:
            if save_dir is None:
                now = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
                save_dir = f"./{self.__class__.__name__} {now}"
            self.save_dir = str(save_dir)
            self.genome_dir = os.path.join(self.save_dir, "genomes")
            os.makedirs(self.genome_dir, exist_ok=True)
        else:
            self.save_dir = None if save_dir is None else str(save_dir)
            self.genome_dir = (
                None if self.save_dir is None else os.path.join(self.save_dir, "genomes")
            )

    # ---------------------------------------------------------- construction

    def _resolve_pop_batch_size(self, pop_batch_size: Optional[int]) -> Optional[int]:
        """Check the pipeline's chunking request against the algorithm's own.

        Args:
            pop_batch_size (int | None): The pipeline's request, or ``None``.

        Returns:
            int | None: The effective chunk size.

        Raises:
            TensorNEATError: If the pipeline asks for a chunk size the algorithm
                was not constructed with (it changes the random stream, so it is
                part of the reproducible configuration).
        """
        declared = getattr(self.algorithm, "pop_batch_size", None)
        if declared is None:
            declared = getattr(self.algorithm, "_pop_batch_size", None)
        if pop_batch_size is None:
            return declared
        if declared is None or int(declared) != int(pop_batch_size):
            raise TensorNEATError(
                f"Pipeline(pop_batch_size={pop_batch_size}) disagrees with the "
                f"algorithm's pop_batch_size={declared!r}. Chunking is fixed when "
                "the algorithm is constructed (it changes the random stream), so "
                "pass it to NEAT(..., pop_batch_size=...) instead of to the pipeline"
            )
        return int(pop_batch_size)

    def _validate_arity(self) -> None:
        """Check that the problem and the algorithm agree on the network arity.

        Raises:
            TensorNEATError: If the problem declares no ``input_shape``, or if
                ``input_shape[-1]``/``output_shape[-1]`` disagree with the
                algorithm's ``num_inputs``/``num_outputs``.
        """
        input_shape = _declared_shape(self.problem, "input_shape")
        if input_shape is None:
            raise TensorNEATError(
                f"problem {type(self.problem).__name__} declares no input_shape; "
                "every problem must expose the per-sample input shape so the "
                "pipeline can check it against algorithm.num_inputs"
            )
        if input_shape[-1] != int(self.algorithm.num_inputs):
            raise TensorNEATError(
                f"algorithm.num_inputs={self.algorithm.num_inputs} does not match "
                f"problem.input_shape={input_shape} (per-sample); the genome and the "
                "problem must agree on the network's input width"
            )
        output_shape = _declared_shape(self.problem, "output_shape")
        if output_shape is not None and output_shape[-1] != int(
            self.algorithm.num_outputs
        ):
            raise TensorNEATError(
                f"algorithm.num_outputs={self.algorithm.num_outputs} does not match "
                f"problem.output_shape={output_shape} (per-sample); the genome and "
                "the problem must agree on the network's output width"
            )

    # --------------------------------------------------------------- config

    def show_config(self) -> dict:
        """Describe the full run configuration as a JSON-serialisable mapping.

        Returns:
            dict: ``{"pipeline": ..., "algorithm": ..., "problem": ...}`` with
            plain Python values (a gene, an operator or a transform is rendered by
            ``host_payload._object_config`` as its class path plus its public
            primitive attributes, callables by their qualified name, so no memory
            address ever reaches the file), so
            ``json.dumps(pipeline.show_config(), indent=4)`` always succeeds, two
            identical runs write identical bytes, and the checkpoint header names
            the search law that produced ``log.txt``.

        Notes:
            Host-only, and read defensively: a genome that does not expose a
            static simply omits it instead of failing the checkpoint.
        """
        genome = self._genome()
        algorithm: dict[str, Any] = {"class": type(self.algorithm).__name__}
        species = getattr(self.algorithm, "species", None)
        if species is None:
            # A wrapper algorithm (`HyperNEAT`) owns no species table of its own:
            # the search law lives on the inner `NEAT`, exactly like the genome
            # `_genome()` falls back to.
            species = getattr(getattr(self.algorithm, "neat", None), "species", None)
        for name in (
            "pop_size", "pop_batch_size", "species_size", "max_stagnation",
            "species_elitism", "spawn_number_change_rate", "genome_elitism",
            "survival_threshold", "min_species_size", "compatibility_threshold",
            "species_number_calculate_by",
        ):
            value = _config_value(self.algorithm, name)
            if value is None and name not in ("pop_size", "pop_batch_size"):
                value = _config_value(species, name)
            if value is not None:
                algorithm[name] = value
        if genome is not None:
            algorithm["genome"] = {
                "class": type(genome).__name__,
                **{
                    name: _config_value(genome, name)
                    for name in (
                        "num_inputs", "num_outputs", "max_nodes", "max_conns",
                        "max_in_degree", "max_depth", "key_span", "node_gene",
                        "conn_gene", "mutation", "crossover", "distance",
                        "input_transform", "output_transform",
                    )
                },
            }
        problem_info = {
            "class": type(self.problem).__name__,
            "input_shape": _jsonable(_declared_shape(self.problem, "input_shape")),
            "output_shape": _jsonable(_declared_shape(self.problem, "output_shape")),
        }
        for name in ("error_method", "max_step", "repeat_times"):
            value = _config_value(self.problem, name)
            if value is not None:
                problem_info[name] = value
        return {
            "pipeline": {
                "class": type(self).__name__,
                "seed": self.seed,
                "fitness_target": _jsonable(self.fitness_target),
                "generation_limit": self.generation_limit,
                "pop_size": self.pop_size,
                "pop_batch_size": self.pop_batch_size,
                "backend": self.backend,
                "device": _jsonable(self.device),
                "compile_options": _jsonable(self.compile_options),
                "distance_centre": self.distance_centre,
                "distance_formulation": self.distance_formulation,
                "kernel_mode": self.kernel_mode,
                "is_save": self.is_save,
                "save_dir": _jsonable(self.save_dir),
                "is_detailed": self.is_detailed,
                "show_problem_details": self.show_problem_details,
                "host_payload": self.host_payload,
            },
            "algorithm": algorithm,
            "problem": problem_info,
        }

    # ---------------------------------------------------------------- setup

    def setup(self, state: Optional[State] = None) -> State:
        """Create the initial state: RNG key, build options, population, task.

        Order matters and follows the kernel contract: ``randkey`` first (the
        single source of randomness), then ``backend``/``device``/
        ``compile_options`` (every ``evoxc.build`` below reads them off the
        state) together with the device-derived formulation statics
        ``distance_centre``/``distance_formulation``/``kernel_mode`` (plain
        strings the generation graph bakes through its static-config closure),
        then ``algorithm.setup`` (population, generation counter, species
        table; it advances ``randkey`` once), then ``problem.setup``.

        Device placement: the setup-time graphs are built and run **on CPU**
        (``device=None`` is registered for them), because their only run input
        is the host ``randkey`` and ``evoxc.run`` never transfers across
        devices. A CPU run cannot execute an artifact compiled for the
        accelerator either, so the device-targeting ``compile_options`` entries
        are stripped for that phase (:meth:`_host_build_options`), and a backend
        that owns no CPU device at all -- ``"xla"``, whose PJRT plugin follows
        the installed platform -- runs that phase on the reference interpreter
        (:meth:`_host_backend`). The host phase is orchestration only: its
        ``backend`` entry is the host backend, and after ``problem.setup`` the
        requested ``backend`` is registered with the real ``device`` and
        ``compile_options`` and the runtime tensors are placed on the device
        (:meth:`_place_state`), so generation 0 and every later :meth:`step` are
        fully device-resident and use the requested backend. With ``device=None``
        nothing is placed and the state is returned with its configuration
        unchanged.

        Args:
            state (State, optional): Base state to extend; ``None`` -- spelled
                this way because a ``State()`` default argument would be
                evaluated at import time -- starts from an empty state.

        Returns:
            State: The initial state, ready for :meth:`step` / :meth:`auto_run`.

        Raises:
            TensorNEATError: Propagated from ``algorithm.setup`` /
                ``problem.setup``; ``common.errors.ConfigError`` if the incoming
                state already carries one of the registered keys.
            evoxc.DeviceError: Propagated from :meth:`_place_state` when the
                requested device cannot be reached (nothing falls back to CPU).
        """
        state = State() if state is None else state
        # Re-derive the formulation statics from the device this run actually
        # places on (a caller may reassign ``.device`` between construction and
        # ``setup``), and register them with the build options: they are static
        # strings the graph's closure reads, so they must be on the state before
        # any ``step`` -- and the setup graphs below never need them.
        (
            self.distance_centre,
            self.distance_formulation,
            self.kernel_mode,
        ) = _formulation_statics(self.device)
        state = state.register(randkey=evoxc.random.key(self.seed))
        state = state.register(
            # Setup graphs stay on CPU: ``algorithm.setup`` runs its
            # initialisation graph on ``randkey``, which is host memory, and a
            # CPU run cannot execute an artifact compiled for the accelerator.
            # A backend with no CPU device builds that phase on the reference
            # interpreter; the requested one is registered again below.
            backend=self._host_backend(),
            distance_centre=self.distance_centre,
            distance_formulation=self.distance_formulation,
            kernel_mode=self.kernel_mode,
            device=None,
            compile_options=self._host_build_options(),
        )
        state = self.algorithm.setup(state)
        state = self.problem.setup(state)

        # The requested backend, device and compile options are registered only
        # now, then the runtime tensors are placed once: every later generation
        # runs and stays on that device.
        state = state.update(
            backend=self.backend,
            device=self.device,
            compile_options=self.compile_options,
        )
        if self.device is not None:
            state = self._place_state(state)

        self.fitness_history = []
        self.best_fitness = float("-inf")
        self.best_genome = None
        self.detail_failures = {}
        self._host_generation = None
        self._executables = {}
        self.shape_signature = None
        self.build_seconds = 0.0
        self.builds = 0

        if self.is_save:
            os.makedirs(self.genome_dir, exist_ok=True)
            with open(os.path.join(self.save_dir, "config.txt"), "w") as handle:
                handle.write(json.dumps(self.show_config(), indent=4))
            with open(os.path.join(self.save_dir, "log.txt"), "w") as handle:
                handle.write("Generation,Max,Min,Mean,Std,Cost Time\n")
        return state

    # ----------------------------------------------------------- compile site

    def _genome(self) -> Any:
        """The genome whose statics define the compiled shape, or ``None``.

        ``NEAT`` exposes it directly; ``HyperNEAT`` holds the CPPN genome on its
        inner ``NEAT`` (the payload -- and therefore the compiled signature --
        belongs to the CPPN, never to the fixed substrate).
        """
        genome = getattr(self.algorithm, "genome", None)
        if genome is None:
            genome = getattr(getattr(self.algorithm, "neat", None), "genome", None)
        return genome

    def _shape_signature(self, state: State) -> tuple[Any, ...]:
        """The static signature that keys the executable cache.

        Args:
            state (State): State after ``setup``; its payload supplies fallbacks
                for a genome that does not expose the full static set.

        Returns:
            tuple: ``(pop_size, max_nodes, max_conns, max_in_degree,
            max_depth, key_span)`` for the CPPN/genome payload, followed by
            :func:`~tensorneat.host_payload._substrate_signature`'s entries
            when the algorithm wraps a
            fixed substrate -- a change in any of them forces a rebuild --,
            then :func:`~tensorneat.host_payload._formulation_signature`'s
            ``(distance_centre, distance_formulation, kernel_mode)`` statics
            (so a device/mode change can never reuse an executable formulated for
            the other one), then the effective ``backend`` -- the level spelling
            reads it at trace time (``DefaultGenome._static_sweep``: ``iree``
            keeps the unrolled sweep, ``xla`` takes the data-dependent loop), so
            a backend change must rebuild rather than reuse the other spelling
            -- and last the tiled-evaluate chunk budget of the
            device the graph runs on
            (:func:`~tensorneat.host_payload.eval_chunk_rows`), because that
            budget is unrolled into the graph at trace time: a CPU build must
            never be reused for a device leg (or the other way round).

        Raises:
            TensorNEATError: If ``state`` carries no four-array payload (it did
                not come from :meth:`setup`).
        """
        pop = getattr(state, "pop", None)
        if pop is None or len(tuple(pop)) != 4:
            raise TensorNEATError(
                "Pipeline.step needs the algorithm's population payload on the "
                "state (four arrays); call setup() first, or pass an algorithm "
                "that registers 'pop'"
            )
        genome = self._genome()

        def static(name: str, fallback: int) -> int:
            """Read one genome static, falling back to a shape-derived value."""
            value = getattr(genome, name, None) if genome is not None else None
            return int(value) if value is not None else int(fallback)

        node_keys, _, conn_keys, _ = tuple(pop)
        max_nodes = static("max_nodes", node_keys.shape[1])
        payload_signature = (
            int(self.algorithm.pop_size),
            max_nodes,
            static("max_conns", conn_keys.shape[1]),
            static("max_in_degree", 0),
            static("max_depth", 0),
            static("key_span", 0),
        )
        # The effective backend is baked into the trace (``_static_config`` keeps
        # the registered string) and selects the level spelling, so it belongs in
        # the cache key just like the formulation statics.
        registered = state.registered_keys()
        backend = state.backend if "backend" in registered else self.backend
        return (
            payload_signature
            + _substrate_signature(self.algorithm)
            + _formulation_signature(state)
            + (backend, eval_chunk_rows(self._device_kind(), max_nodes))
        )

    def _host_backend(self) -> str:
        """The backend the CPU-only setup/reporting graphs build on.

        The setup phase registers ``device=None`` for its graphs, so the backend
        it names must be able to serve ``Device("cpu", 0)``. ``"xla"`` cannot:
        it drives a PJRT C API plugin, and the installed plugin's platform (here
        ``rocm``) is the only device kind it serves, so a CPU build raises
        ``evoxc``'s own ``BackendError``. Such a backend therefore hands that
        phase to the reference interpreter (:data:`_HOST_BACKEND_FALLBACKS`);
        every other backend -- ``numpy`` and ``iree``, which compiles an
        llvm-cpu artifact when no device is named -- keeps building its own host
        artifact.

        This is not a silent fallback for the *run*: the host graphs are
        host-side orchestration only (initialisation, schedule derivation,
        reporting), their result tensors are host tensors, and
        :meth:`_place_state` puts them on the requested device afterwards, while
        the requested backend is registered again before generation 0. A backend
        that cannot reach its requested device still raises.

        Returns:
            str: :data:`_HOST_BACKEND_FALLBACKS`'s entry for :attr:`backend`, or
            :attr:`backend` itself.
        """
        return _HOST_BACKEND_FALLBACKS.get(self.backend, self.backend)

    def _host_build_options(self) -> Optional[dict]:
        """The compile options the CPU-only setup graphs may see.

        ``algorithm.setup`` / ``problem.setup`` build and run one-off graphs on
        the host ``randkey`` (the state registers ``device=None`` for that
        phase), so the options that select *where* a compiled artifact runs --
        ``"device"`` and ``"target_backends"`` -- must not be forwarded: an
        artifact compiled for the accelerator cannot execute on the CPU device
        it is run on. Every unrelated key (``iree_compile_args``,
        ``iree_runtime_args``, flags, ...) is preserved, and ``None`` stays
        ``None``.

        When the host phase runs on a *different* backend than the requested one
        (:meth:`_host_backend` -- the reference interpreter for ``"xla"``), the
        whole mapping is dropped: those keys describe the requested backend's
        device and compiler (``plugin_path``, ``target_backends``, compiler
        flags), and the reference interpreter must not receive options it does
        not recognize.
        """
        if not self.compile_options:
            return self.compile_options
        if self._host_backend() != self.backend:
            return None
        return {
            key: value
            for key, value in self.compile_options.items()
            if key not in _DEVICE_TARGETING_OPTIONS
        }

    def _build_options(self, state: Optional[State] = None) -> dict:
        """The ``evoxc.build`` keyword arguments this pipeline forwards.

        ``backend`` always, ``device`` only when it is not ``None``, and every
        ``compile_options`` entry expanded into ``**options``. Nothing is
        defaulted or rewritten: a backend that cannot compile a kernel raises its
        own error instead of falling back.

        ``state`` selects *whose* build selection is forwarded -- its
        ``backend``/``device``/``compile_options`` entries win when registered,
        :attr:`backend`/:attr:`device`/:attr:`compile_options` otherwise: the
        generation graph passes the real state, whose entries are the requested
        backend and device (:meth:`setup` registers them before generation 0),
        while a host-reporting call passes :meth:`_host_view`'s state, whose
        entries are the CPU host phase -- ``device=None`` plus the host backend
        (:meth:`_host_backend`) and its options (:meth:`_host_build_options`) --
        so reports build and run on the CPU side-channel instead of paying a
        fresh compile on the requested device.
        """
        registered = state.registered_keys() if state is not None else ()
        backend = state.backend if "backend" in registered else self.backend
        device = state.device if "device" in registered else self.device
        compile_options = (
            state.compile_options
            if "compile_options" in registered
            else self.compile_options
        )
        options: dict[str, Any] = {"backend": backend}
        if device is not None:
            options["device"] = device
        if compile_options:
            options.update(dict(compile_options))
        return options

    def _placement_device(self) -> Optional[Any]:
        """The ``evoxc.Device`` the runtime tensors are placed on, or ``None``.

        ``None`` means "leave everything where the backend put it" (the
        default CPU path). A kind string is resolved to
        ``evoxc.Device(kind)`` -- ``"cpu"`` maps to ``Device("cpu", 0)``, whose
        ``Tensor.to`` is the identity -- and an ``evoxc.Device`` is returned
        unchanged. Any other value is handed to ``Tensor.to`` as-is so
        ``evoxc``'s own ``TypeError`` names the problem.
        """
        if self.device is None:
            return None
        if isinstance(self.device, str):
            return evoxc.Device(self.device)
        return self.device

    def _device_kind(self) -> Optional[str]:
        """``.kind`` of the device the generation graph runs on (``None`` for CPU).

        The single spelling of "which leg is this?", used by the two places that
        must agree about it: the tiled-evaluate chunk budget the evaluate pass
        unrolls at trace time (:func:`~tensorneat.host_payload.eval_chunk_rows`)
        and the executable-cache signature that records it
        (:meth:`_shape_signature`). ``device=None`` and ``device="cpu"`` both
        resolve to the interpreter's own CPU budget, so the default path and an
        explicit CPU placement stay byte-identical.

        Returns:
            str | None: ``"cpu"`` for :meth:`_placement_device`'s
            ``Device("cpu", 0)``, the kind string of an accelerator device, or
            ``None`` when no device was requested (the graph then runs wherever
            the backend puts it -- on llvm-cpu for ``backend="iree"``).
        """
        return getattr(self._placement_device(), "kind", None)

    def _place_state(self, state: State) -> State:
        """Place every runtime tensor on :meth:`_placement_device`'s device.

        Only the :data:`_RUNTIME_KEYS` entries are mapped -- the ``State`` also
        holds configuration objects and the ``compile_options``/``device``
        host values, which must never be walked -- and every non-tensor leaf is
        left alone.

        A cross-kind move goes through host memory: the runtime tensors
        ``algorithm.setup`` returns are payload-backed (one graph run), and
        ``evoxc`` refuses a payload-to-device transfer ("cross-device copies are
        not implemented in v1"). The supported upload path is an
        *ndarray-backed* tensor, so such a leaf is read back with the explicit
        D2H accessor (:func:`~tensorneat.common.tools.to_numpy`) and
        re-uploaded as contiguous memory. Contiguity is fixed **rank-preservingly**
        -- ``np.ascontiguousarray`` promotes a rank-0 array to shape ``(1,)``,
        which would break the scalar ``randkey`` (``evoxc.random.split`` rejects
        it) -- so only a genuinely non-contiguous array is copied. A same-kind
        move stays ``Tensor.to``, which returns ``self`` on the tensor's own
        device, so the call is idempotent, a CPU placement copies nothing, and
        the default ``device=None`` path returns the state untouched.

        Uploading goes through the *requested* backend's placement provider:
        ``Tensor.to`` dispatches to the preferred transfer backend's provider
        for the target kind, and that preference is recorded by the backend
        registry on lookup, so :attr:`backend` is resolved first -- a registry
        lookup only (no plugin load, no compile) -- instead of letting another
        adapter's flat default provider supply a payload the requested backend's
        executables would reject. A backend with no provider for that kind
        (``"numpy"``) leaves the registry preference untouched.
        """
        device = self._placement_device()
        if device is None:
            return state
        evoxc.backends.get(self.backend)

        def place(leaf: Any) -> Any:
            """Move one tensor leaf to ``device``; leave host values alone."""
            if not isinstance(leaf, evoxc.Tensor):
                return leaf
            source = getattr(leaf, "device", None)
            if source is not None and source.kind != device.kind:
                host = np.asarray(to_numpy(leaf))
                if not host.flags["C_CONTIGUOUS"]:
                    host = np.ascontiguousarray(host)
                return evoxc.from_numpy(host).to(device)
            return leaf.to(device)

        placed = {
            name: evoxc.tree_map(place, getattr(state, name))
            for name in _RUNTIME_KEYS
            if name in state.registered_keys()
        }
        return state.update(**placed)

    def _host_view(self, state: State) -> State:
        """A CPU view of ``state`` for the host-reporting graphs.

        Reporting graphs (``_schedule_of``, ``_constant_act_fn``, ``show`` and
        the two ``show_details`` hooks) consume host data (``evoxc.from_numpy``
        / ``evoxc.constant`` are CPU-only), so they keep building and running on
        the CPU side-channel: this view registers ``device=None`` (its build
        options carry no device) together with the host phase's backend and
        options (:meth:`_host_backend` / :meth:`_host_build_options` -- the
        reference interpreter for a backend that owns no CPU device), and moves
        the runtime tensors to ``Device("cpu", 0)``. That move is evoxc's
        explicit D2H transfer (``Tensor.to(cpu)`` materialises a fresh host copy
        of a device payload), so a device-resident state is never rejected and
        the result is never fed back into evolution. The default ``device=None``
        path returns ``state`` itself.
        """
        if self.device is None:
            return state
        cpu = evoxc.Device("cpu", 0)

        def to_host(leaf: Any) -> Any:
            """Move one tensor leaf to the host; leave host values alone."""
            return leaf.to(cpu) if isinstance(leaf, evoxc.Tensor) else leaf

        placed = {
            name: evoxc.tree_map(to_host, getattr(state, name))
            for name in _RUNTIME_KEYS
            if name in state.registered_keys()
        }
        backend = (
            self._host_backend()
            if "backend" in state.registered_keys()
            else self.backend
        )
        compile_options = (
            self._host_build_options()
            if "compile_options" in state.registered_keys()
            else self.compile_options
        )
        return state.update(
            backend=backend, device=None, compile_options=compile_options, **placed
        )

    def _static_config(self, state: State) -> State:
        """The tensor-free configuration ``State`` baked into the graph.

        Every registered entry survives except the runtime tensors -- the
        :data:`_RUNTIME_KEYS` names and any remaining ``evoxc.Tensor`` -- which
        the kernel rebuilds from its tensor arguments. Baking a concrete tensor
        into a trace would freeze the current generation's data.
        """
        entries = {}
        for name in state.registered_keys():
            if name in _RUNTIME_KEYS:
                continue
            value = getattr(state, name)
            if isinstance(value, evoxc.Tensor):
                continue
            entries[name] = value
        entries["pop_size"] = int(self.algorithm.pop_size)
        return State(**entries)

    def _generation_kernel(
        self,
        config: State,
        species_tree: Any,
        device_kind: Optional[str] = None,
    ) -> Callable:
        """Build the traced function of one whole generation.

        Args:
            config (State): Tensor-free configuration
                (:meth:`_static_config`), captured by the closure so the graph
                inputs stay pure tensors.
            species_tree (Any): ``evoxc.TreeSpec`` of the ``SpeciesState``
                pytree, used to rebuild it inside the trace from its leaves.
            device_kind (str | None, optional): ``.kind`` of the device the graph
                runs on (:meth:`_device_kind`), threaded into the ``act_fn`` so
                the evaluate pass picks its chunk budget at trace time.
                Defaults to ``None`` (the CPU interpreter).

        Returns:
            Callable: ``kernel(randkey, node_keys, node_attrs, conn_keys,
            conn_attrs, generation, *species_leaves)`` returning the flat list
            ``[randkey, generation, *pop, *species, packed]``.
        """
        algorithm = self.algorithm
        problem = self.problem
        species_leaves = int(species_tree.num_leaves)

        def kernel(
            randkey: Any,
            node_keys: Any,
            node_attrs: Any,
            conn_keys: Any,
            conn_attrs: Any,
            generation: Any,
            *species_values: Any,
        ) -> list:
            """One generation: ``split`` -> ``ask`` -> ``transform`` -> ``evaluate`` -> ``tell``.

            The arguments are the four split-payload arrays (``(P, N) int32``
            keys with ``I_INF`` padding, ``(P, N, NA) float32`` attrs with NaN
            padding, ``(P, E, CK) int32`` and ``(P, E, CA) float32``), the 0-d
            float32 generation counter and the ``SpeciesState`` leaves.

            The payload registered on ``current`` is the graph's own argument;
            ``algorithm.ask(current)`` then supplies the population that
            ``transform`` and ``evaluate`` see.  ``tell`` receives ``current``,
            whose ``pop`` is still the argument payload, so the returned
            ``next_state.pop`` is bred from it regardless of ``ask``.

            Returns:
                list: ``[randkey, generation, *next_pop, *next_species, packed]``,
                where ``packed`` is the one ``(P + 2,)`` host block described in
                :meth:`step`.

            Raises:
                TensorNEATError: If ``tell`` changes the species leaf count,
                    which would break the host-side rebuild.
            """
            pop = GenomeTensors(node_keys, node_attrs, conn_keys, conn_attrs)
            species = evoxc.tree_unflatten(list(species_values), species_tree)
            eval_key, tell_key = evoxc.random.split(randkey)
            current = config.register(
                pop=pop, generation=generation, species=species, randkey=tell_key
            )
            # ``ask`` is the algorithm's "population to evaluate" hook and belongs
            # in the graph: a shipped ``NEAT.ask``/``HyperNEAT.ask`` returns
            # ``state.pop`` unchanged (zero ops, bit-identical), while an
            # algorithm that overrides ``ask`` is honoured instead of bypassed.
            # Its result drives ``transform`` and the problem's evaluate only:
            # ``tell`` still breeds from ``current``, and the host-side
            # ``previous_pop``/checkpoints report ``state.pop``, so a non-identity
            # ``ask`` must document that reporting mismatch.
            pop = algorithm.ask(current)
            transformed = algorithm.transform(current, pop)
            act_fn = _NetworkActFn(
                lambda: (current, transformed, pop),
                algorithm.forward_prepared,
                algorithm.prepare_forward,
                device_kind,
            )
            fitness = problem.evaluate(current, eval_key, act_fn, transformed)
            next_state = algorithm.tell(current, fitness)
            overflow = _schedule_overflow(transformed)
            max_overflow = (
                enp.full((), 0, dtype=evoxc.int32)
                if overflow is None
                else evoxc.max(overflow)
            )
            next_species = evoxc.tree_leaves(next_state.species)
            if len(next_species) != species_leaves:
                raise TensorNEATError(
                    "algorithm.tell changed the species pytree: "
                    f"{species_leaves} leaves in, {len(next_species)} out; the "
                    "pipeline rebuilds the state from a fixed leaf layout"
                )
            # One leaf instead of three: the fitness vector plus the two scalars the
            # host bookkeeping needs, in the fitness dtype so no mixed-dtype
            # ``concatenate`` is exported.  A device-to-host copy costs a fixed
            # ~50 us whatever its size, so three leaves would be three syncs where
            # this is one; both counters are exact in a float32 at any generation
            # count a run can reach (the state already stores ``generation`` as
            # float32, so packing cannot lose precision the state does not).
            packed = evoxc.concatenate(
                [
                    fitness,
                    evoxc.reshape(
                        evoxc.cast(next_state.generation, fitness.dtype), (1,)
                    ),
                    evoxc.reshape(evoxc.cast(max_overflow, fitness.dtype), (1,)),
                ],
                axis=0,
            )
            return [
                next_state.randkey,
                next_state.generation,
                *next_state.pop,
                *next_species,
                packed,
            ]

        return kernel

    def _build_executable(self, state: State) -> Any:
        """Trace, lower, compile and load the generation graph for ``state``.

        Args:
            state (State): State after ``setup``; its tensors define both the
                input specs and the static configuration.

        Returns:
            Any: The loaded ``evoxc.Executable``.

        Raises:
            evoxc.EVOXCError subclasses: Propagated unchanged from
                ``evoxc.build`` -- nothing here retries elsewhere.
        """
        kernel = self._generation_kernel(
            self._static_config(state),
            evoxc.tree_structure(state.species),
            self._device_kind(),
        )
        specs = [
            evoxc.TensorSpec(tuple(state.randkey.shape), state.randkey.dtype),
            *(evoxc.TensorSpec(tuple(field.shape), field.dtype) for field in state.pop),
            evoxc.TensorSpec(tuple(state.generation.shape), state.generation.dtype),
            *(
                evoxc.TensorSpec(tuple(leaf.shape), leaf.dtype)
                for leaf in evoxc.tree_leaves(state.species)
            ),
        ]
        return evoxc.build(kernel, *specs, **self._build_options(state))

    def _executable_for(self, state: State) -> Any:
        """The cached executable for ``state``'s shape signature.

        Built on first use and then reused; the build is timed into
        :attr:`build_seconds` and counted in :attr:`builds`. A *different*
        signature (population or genome geometry, substrate geometry, the
        trace-time formulation statics, the effective backend or the evaluate
        chunk budget) is a second entry, so a shape, device/mode or backend
        change can never silently reuse the previous graph.
        """
        signature = self._shape_signature(state)
        self.shape_signature = signature
        cached = self._executables.get(signature)
        if cached is not None:
            return cached
        start = time.perf_counter()
        executable = self._build_executable(state)
        self.build_seconds = time.perf_counter() - start
        self.builds += 1
        self._executables[signature] = executable
        return executable

    # ------------------------------------------------------------- one step

    def step(
        self,
        state: State,
        host_payload: Optional[bool] = None,
    ) -> tuple[State, Optional[GenomeTensors], np.ndarray]:
        """Advance the search by exactly one generation (one compiled graph run).

        Args:
            state (State): Current state from :meth:`setup` or the previous
                :meth:`step`.
            host_payload (bool | None, optional): Whether to copy the evaluated
                population to host numpy arrays. ``None`` (the default) means
                "whatever the constructor's ``host_payload`` said"; ``True`` or
                ``False`` overrides it for this one call, which is how
                :meth:`auto_run` asks for the payload only on the generations it
                reports or checkpoints.

        Returns:
            tuple: ``(new_state, previous_pop, fitnesses)`` -- the state of the
            *next* generation (advanced ``randkey``, incremented ``generation``,
            new population, updated species), the gene payload that was
            evaluated (population on axis 0) and the ``(P,)`` host float fitness,
            **fitness last**.

            With ``host_payload`` in effect the payload is four **host numpy**
            arrays (``node_keys (P, max_nodes)``, ``node_attrs (P, max_nodes, A)``,
            ``conn_keys (P, max_conns, key_width)``, ``conn_attrs (P, max_conns,
            W)``). Without it (``Pipeline(host_payload=False)`` and no override)
            the payload is handed back **as the backend left it** -- numpy on the
            reference interpreter, device-resident ``evoxc.Tensor`` leaves on a
            device backend -- so no device-to-host copy is paid; materialise a
            field with :func:`~tensorneat.common.tools.to_numpy` when a host
            array is wanted. The fitness vector is always host: the generation
            graph returns it packed together with the two scalars the loop needs
            (see below), and that block is one copy.

        Raises:
            TensorNEATError: If the schedule overflowed (``edge_overflow > 0``):
                some connections did not fit the static ``max_in_degree`` window,
                so the evaluated phenotype would silently differ from the genome.
            evoxc.ShapeError / evoxc.DTypeError: Propagated when the state does
                not match the cached signature -- shapes are static, and a
                mismatch is never reshaped away.

        Notes:
            Prints nothing (:meth:`auto_run` owns all output).  The generation
            graph calls ``algorithm.ask``: every shipped ``ask`` returns
            ``state.pop`` unchanged (zero ops), so ``state.pop`` *is* the
            population that produced ``fitnesses`` and it is returned as the
            evaluated payload.  An algorithm whose ``ask`` returns a different
            population is honoured for ``transform``/``evaluate``, but the
            returned ``previous_pop``, the checkpoints and ``tell`` still
            describe ``state.pop`` -- the documented reporting mismatch.

            The graph returns the fitness vector and the two host-side scalars
            (the incremented generation and the schedule's overflow maximum) as
            one ``(P + 2,)`` block, so a generation pays **one** device-to-host
            copy for all three instead of one per leaf; the generation number is
            kept on :attr:`_host_generation` for the report.  The overflow test
            raises before anything is returned, exactly as before.
        """
        materialize = self.host_payload if host_payload is None else bool(host_payload)
        executable = self._executable_for(state)
        species_leaves = evoxc.tree_leaves(state.species)
        outputs = evoxc.run(
            executable, state.randkey, *state.pop, state.generation, *species_leaves
        )
        split = 6 + len(species_leaves)
        packed = np.asarray(to_numpy(outputs[split]))
        fitnesses = packed[: self.pop_size]
        max_overflow = int(packed[self.pop_size + 1])
        if max_overflow > 0:
            raise TensorNEATError(
                f"the execution schedule dropped {max_overflow} connection(s): a "
                "node had more incoming edges than the static window "
                f"max_in_degree={getattr(self._genome(), 'max_in_degree', None)} can "
                f"hold (population {self.pop_size}). max_in_degree is a genome "
                "static, so rebuild the genome with a larger max_in_degree -- the "
                "pipeline will not widen it silently, and a lossy schedule is never "
                "used"
            )
        self._host_generation = int(packed[self.pop_size])
        new_state = state.update(
            pop=GenomeTensors(*outputs[2:6]),
            generation=outputs[1],
            species=evoxc.tree_unflatten(
                list(outputs[6:split]), evoxc.tree_structure(state.species)
            ),
            randkey=outputs[0],
        )
        if materialize:
            previous_pop = GenomeTensors(*(to_numpy(field) for field in state.pop))
        else:
            # No host copy: hand the payload back exactly as the backend left it.
            # The reporting paths materialise it themselves when they need it.
            previous_pop = GenomeTensors(*tuple(state.pop))
        return new_state, previous_pop, fitnesses

    # ------------------------------------------------------------- run loop

    def auto_run(self, state: State) -> tuple[State, Optional[GenomeTensors]]:
        """Run generations until the fitness target or the generation limit.

        Args:
            state (State): Initial state from :meth:`setup`.

        Returns:
            tuple: ``(state, best_genome)`` -- the final state and the best
            payload seen, a ``GenomeTensors`` of host numpy arrays shaped as a
            **one-individual batch** (``node_keys (1, max_nodes)``,
            ``node_attrs (1, max_nodes, A)``, ``conn_keys (1, max_conns,
            key_width)``, ``conn_attrs (1, max_conns, W)`` -- ``state.pop`` at
            ``P = 1``), or ``None`` if no generation produced a finite fitness.

        Raises:
            TensorNEATError: Propagated from :meth:`step` (e.g. schedule
                overflow).
            evoxc.EVOXCError subclasses: Propagated from the build/run.

        Notes:
            The executable is built once here (timed into
            :attr:`build_seconds`) and reused for every generation; the host loop
            is what makes logging, checkpointing and early stopping possible
            without putting Python control flow into the graph.  The early stop
            compares :func:`_finite_fitness`'s view, so a non-finite fitness
            (``+inf``/``NaN`` from a custom problem) never satisfies
            ``fitness_target``; ``-inf`` is the only non-finite value selection
            ever sees.

            The host payload is requested from :meth:`step` only when this loop
            can consume it -- the constructor's ``host_payload`` (the default), a
            checkpoint to write, the per-generation report or the problem's
            ``show_details`` hook.  With ``Pipeline(host_payload=False)``,
            ``is_detailed=False`` and no saving, a run therefore never pays a
            per-generation payload copy for its caller; it still records
            :attr:`fitness_history` and :attr:`best_fitness`.  This loop's own
            best-genome bookkeeping does materialise the payload whenever a
            generation sets a new best -- the first finite generation always
            does -- so :attr:`best_genome` is populated rather than ``None``,
            and only those generations pay the copy.
        """
        self._executable_for(state)
        if self.is_detailed:
            print(f"compile finished, cost time: {self.build_seconds:.6f}s")

        # The payload is only worth a device-to-host copy when something below
        # consumes it: the opt-out flag itself, a checkpoint, the report or the
        # problem's hook.
        wants_payload = (
            self.host_payload
            or self.is_save
            or self.is_detailed
            or self.show_problem_details
        )
        for _ in range(self.generation_limit):
            start = time.perf_counter()
            state, previous_pop, fitnesses = self.step(
                state, host_payload=wants_payload
            )
            self._analyse(
                state,
                previous_pop,
                fitnesses,
                time.perf_counter() - start,
                self._host_generation,
            )
            if float(np.max(_finite_fitness(fitnesses))) >= self.fitness_target:
                if self.is_detailed:
                    print("Fitness limit reached!")
                break
        else:
            # The generation number comes from the last step's packed host block
            # (no second device read); a zero-generation run has none, so fall
            # back to the state it was given.
            generation = (
                self._host_generation
                if self._host_generation is not None
                else int(to_numpy(state.generation))
            )
            if self.is_detailed and generation >= int(self.generation_limit):
                print("Generation limit reached!")

        if self.is_save:
            self._save_best()
        return state, self.best_genome

    def _analyse(
        self,
        state: State,
        previous_pop: Optional[GenomeTensors],
        fitnesses: np.ndarray,
        cost_time: float,
        generation: Optional[int] = None,
    ) -> None:
        """Record one generation on the host: history, best, log and report.

        ``state`` is the state *after* the step, ``previous_pop``/``fitnesses``
        are the population that produced this generation's fitness and
        ``cost_time`` its wall time in seconds.  ``generation`` is the host
        number :meth:`step` read out of its packed host block (so this method
        never pays a second device read for a scalar); ``None`` falls back to the
        state's own counter, which only costs a copy for a caller that skipped
        :meth:`step`.
        The best individual is stored **batch-first** -- the selected rows with a
        leading population axis of 1 added back, so :attr:`best_genome` is
        ``state.pop`` at ``P = 1``, never a set of bare rows.
        ``previous_pop`` may be the backend's own (possibly device-resident)
        payload when ``Pipeline(host_payload=False)`` told :meth:`step` not to
        materialise it; the host arrays are then produced here, **only** on the
        generations that need them -- a new best, a checkpoint, or the problem's
        ``show_details`` hook -- and a reporting path that needs them but was
        handed ``None`` raises a :class:`~tensorneat.common.errors.TensorNEATError`
        naming the flag instead of silently reporting nothing.
        Both diagnostics hooks are *reporting* only, so neither can abort an
        evolution run: a failing hook is printed (never swallowed silently), the
        failure is recorded in :attr:`detail_failures` and the loop continues.
        Statistics and best-individual selection read :func:`_finite_fitness`'s
        view, so a ``NaN``/``+inf`` fitness from a custom problem is logged as a
        non-finite (excluded) value and can never become :attr:`best_fitness`.
        """
        if generation is None:
            generation = int(to_numpy(state.generation))
        valid = fitnesses[np.isfinite(fitnesses)]
        if valid.size == 0:
            max_f = min_f = mean_f = std_f = float("nan")
        else:
            max_f = float(np.max(valid))
            min_f = float(np.min(valid))
            mean_f = float(np.mean(valid))
            std_f = float(np.std(valid))
        self.fitness_history.append((generation, float(max_f)))

        clean = _finite_fitness(fitnesses)
        max_index = int(np.argmax(clean))
        # The host payload is charged once per generation it is actually needed:
        # a new best, a checkpoint, or the problem's hook.  Everything else in
        # this method reads ``fitnesses`` alone.
        selected = float(clean[max_index])
        needs_best = selected > self.best_fitness
        needed = needs_best or self.is_save or self.show_problem_details
        payload = materialize_payload(previous_pop) if needed else None
        if needs_best:
            self.best_fitness = selected
            # ``payload``'s fields are ``(P, ...)``, so an index gives a bare row;
            # add the leading axis back to keep the one-individual batch.
            self.best_genome = GenomeTensors(
                *(np.asarray(field[max_index])[np.newaxis, ...] for field in payload)
            )

        if self.is_save:
            self._save_generation(generation, payload, clean, max_index)
            with open(os.path.join(self.save_dir, "log.txt"), "a") as handle:
                handle.write(
                    f"{generation},{max_f},{min_f},{mean_f},{std_f},{cost_time}\n"
                )

        # Reporting reads the CPU side-channel (:meth:`_host_view`): the hooks
        # build their own graphs from host data, and a device-resident state
        # would force a fresh compile on the requested device per generation.
        # The view is built only when a hook will actually run.
        report_state = (
            self._host_view(state)
            if self.is_detailed or self.show_problem_details
            else state
        )

        if self.is_detailed:
            print(
                f"Generation: {generation}, Cost time: {cost_time * 1000:.2f}ms\n",
                f"\tfitness: valid cnt: {valid.size}, max: {max_f:.4f}, "
                f"min: {min_f:.4f}, mean: {mean_f:.4f}, std: {std_f:.4f}\n",
            )
            self._details(
                "algorithm.show_details",
                lambda: self.algorithm.show_details(report_state, fitnesses),
            )

        if self.show_problem_details:
            self._details(
                "problem.show_details",
                lambda: self.problem.show_details(
                    report_state,
                    report_state.randkey,
                    self._constant_act_fn(report_state, payload),
                    self._schedule_of(report_state, payload),
                ),
            )

    def _details(self, label: str, callback: Callable[[], Any]) -> None:
        """Run one diagnostics hook, reporting a failure instead of aborting.

        The failure is printed with its own message and its type/message is kept
        in :attr:`detail_failures` (first failure per hook, so a hook that breaks
        every generation cannot grow the record without bound).  That is what
        keeps this different from swallowing the error: the failure stays
        visible in the run log *and* inspectable on the pipeline, while the
        evolution loop itself is unaffected.
        """
        try:
            callback()
        except Exception as error:  # noqa: BLE001 - diagnostics must not abort a run
            self.detail_failures.setdefault(label, (type(error).__name__, str(error)))
            print(
                f"{label} failed ({type(error).__name__}: {error}); continuing the run"
            )

    # ------------------------------------------------------- host-side show

    def _schedule_of(self, state: State, payload: GenomeTensors) -> Any:
        """Derive a host payload's execution schedule, outside any trace.

        The schedule derivation is graph code (``argsort``/``gather``), so it
        runs in a one-off executable whose inputs are ``payload``'s four arrays;
        its result is that algorithm's schedule (``Transformed`` or a wrapper)
        with concrete ``evoxc.Tensor`` leaves.
        """
        config = self._static_config(state)
        algorithm = self.algorithm
        fields = tuple(payload)

        def kernel(*arrays: Any) -> Any:
            """Rebuild the payload from graph data and derive its schedule."""
            return algorithm.transform(config, GenomeTensors(*arrays))

        executable = evoxc.build(
            kernel,
            *(evoxc.TensorSpec(tuple(f.shape), f.dtype) for f in fields),
            **self._build_options(state),
        )
        return evoxc.run(
            executable, *(evoxc.from_numpy(np.ascontiguousarray(f)) for f in fields)
        )

    def _constant_act_fn(self, state: State, payload: GenomeTensors) -> _NetworkActFn:
        """An ``act_fn`` that carries its payload as graph constants.

        Its ``source()`` rebuilds ``payload`` with ``evoxc.constant`` and re-derives
        the schedule **per call**, so it is valid inside *any* trace -- including the
        one a problem builds for its ``show`` report -- because a concrete tensor
        captured in a closure would raise ``evoxc.TraceError`` instead.
        A one-shot caller (func-fit ``show``) therefore behaves exactly as before,
        while a ``prepare``-based caller derives the payload, state and schedule once.

        ``payload`` is taken **as given** -- already batch-first at its own
        population size: :meth:`show` passes one individual, ``_analyse`` the whole
        population.
        """
        config = self._static_config(state)
        algorithm = self.algorithm
        arrays = tuple(np.ascontiguousarray(to_numpy(field)) for field in payload)

        def source() -> tuple[State, Any, GenomeTensors]:
            """Bake the payload as constants and derive its schedule in this trace.

            Returns:
                ``(state, transformed, payload)`` with the payload as graph constants.
            """
            one = GenomeTensors(
                *(evoxc.constant(evoxc.from_numpy(array)) for array in arrays)
            )
            network = config.register(pop=one)
            return network, algorithm.transform(network, one), one

        return _NetworkActFn(
            source, algorithm.forward_prepared, algorithm.prepare_forward
        )

    def show(self, state: State, best: Any) -> None:
        """Print the behaviour of one genome.

        Args:
            state (State): State carrying the configuration (a ``setup`` output).
            best (Any): The genome to report, spelled **batch-first** like
                :attr:`best_genome`: a ``GenomeTensors`` of four arrays with a
                leading population axis of 1 (``node_keys (1, max_nodes)``,
                ``conn_keys (1, max_conns, key_width)``, ...), a four-element
                sequence of those, or the ``best_genome`` :meth:`auto_run`
                returns.  The bare per-individual rows a checkpoint stores are
                accepted too and promoted to that batch.

        Returns:
            None; the report comes from the problem's ``show`` (stdout).

        Raises:
            TensorNEATError: If ``best`` is missing (``None`` -- e.g. a run with no
                finite fitness), is not a four-array genome payload, carries more
                than one individual, or arrays disagreeing about the population
                axis; see ``host_payload._one_individual`` for the two spellings.

        Notes:
            Host-side only. The schedule is derived once for the report (so a
            problem that leans on it gets a real one) and the ``act_fn`` handed to
            the problem rebuilds the same genome from ``evoxc.constant`` values.
        """
        payload = GenomeTensors(*_one_individual(best))
        # Reports read the CPU side-channel: the problem's ``show`` graph bakes
        # host constants (``evoxc.constant`` is CPU-only) and carries its own
        # ``build_options``, so it must see ``device=None``.
        report_state = self._host_view(state)
        return self.problem.show(
            report_state,
            report_state.randkey,
            self._constant_act_fn(report_state, payload),
            self._schedule_of(report_state, payload),
        )

    # --------------------------------------------------------- checkpointing

    def _save_generation(
        self,
        generation: int,
        previous_pop: GenomeTensors,
        fitnesses: np.ndarray,
        max_index: int,
    ) -> None:
        """Write ``genomes/<generation>.npz`` -- this generation's best genome.

        Plain ``numpy.savez`` of the best individual's four **bare** payload rows
        (one genome per file: no population axis), plus its own fitness, so
        ``np.load`` round-trips to a payload :meth:`show` accepts.
        """
        np.savez(
            os.path.join(self.genome_dir, f"{generation}.npz"),
            node_keys=np.asarray(previous_pop.node_keys[max_index]),
            node_attrs=np.asarray(previous_pop.node_attrs[max_index]),
            conn_keys=np.asarray(previous_pop.conn_keys[max_index]),
            conn_attrs=np.asarray(previous_pop.conn_attrs[max_index]),
            fitness=float(fitnesses[max_index]),
        )

    def _save_best(self) -> None:
        """Write ``genomes/best_genome.npz`` for the best genome of the run.

        :attr:`best_genome` is a one-individual batch, so the leading population
        axis is stripped and the file holds the four **bare** rows of a single
        genome -- one genome per file, exactly like ``genomes/<generation>.npz``.
        A no-op when no generation produced a finite fitness: there is nothing to
        checkpoint, and an empty file would look like a valid genome.
        """
        if self.best_genome is None:
            return
        np.savez(
            os.path.join(self.genome_dir, "best_genome.npz"),
            node_keys=np.asarray(self.best_genome.node_keys)[0],
            node_attrs=np.asarray(self.best_genome.node_attrs)[0],
            conn_keys=np.asarray(self.best_genome.conn_keys)[0],
            conn_attrs=np.asarray(self.best_genome.conn_attrs)[0],
            fitness=float(self.best_fitness),
        )
