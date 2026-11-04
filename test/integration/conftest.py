"""Shared fixtures and helpers of the ``./integration`` suite: one tiny ``Pipeline`` plus the host-side drivers.
Every test in this directory drives a whole ``tensorneat.Pipeline`` on the numpy backend, so the shared layer covers exactly the things a whole-pipeline test needs and nothing else.
``tn``, ``payloads`` and ``runner`` are resolved through :func:`support.lazy.require`, which imports them at fixture time and reports a pending implementation as a named skip instead of turning the module into a collection error.
:data:`TINY_CONFIG` is the single small XOR configuration every pipeline in this suite is built from; :func:`tiny_pipeline` spells it out and refuses unknown knobs instead of ignoring them.
:func:`run_pipeline` is ``setup()`` plus ``auto_run()``; the per-generation trajectory comes from :func:`step_loop`, which drives the documented ``pipeline.step`` one generation at a time and never reads an internal history attribute.
:func:`evaluate_population` re-scores any gene payload on the host through the documented kernel triple ``algorithm.transform`` → replicated ``algorithm.forward`` → ``problem.evaluate``.
:func:`missing_evoxc_pipeline_features` and :func:`require_evoxc_pipeline` are a structural probe of the loaded ``tensorneat.pipeline``: while that module is still the legacy JAX file the whole suite skips, and the skip disappears by itself once the evoxc compile site lands.
:func:`record_compiles` counts the calls into the evoxc compile entry points, which is how a test proves that a second generation of the same shape does not recompile.
No ``jax``, no ``torch``, no network, no GPU, no wall-clock assertion: the backend is ``"numpy"`` everywhere and a run is judged by generations, fitness and payload shapes.
"""

from __future__ import annotations

import contextlib
import inspect
import sys
from pathlib import Path
from typing import Any, Callable, Iterator

import numpy as np
import pytest

#: The knobs :func:`tiny_pipeline` starts from, grouped by the object that owns them.
#:
#: Genome knobs (``tensorneat.genome.DefaultGenome``): ``num_inputs``/``num_outputs``
#: are XOR's per-sample shapes, ``max_nodes``/``max_conns``/``max_in_degree``/
#: ``max_depth`` are the static shape signature the Pipeline caches executables
#: by, and ``output_transform`` bounds the read-out so the target is reachable
#: (the name is resolved against the ``ACT`` registry; pass a callable or ``None``
#: to override it).
#: NEAT knobs (``tensorneat.algorithm.NEAT``): ``pop_size`` and ``species_size``;
#: ``species_size * min_species_size <= pop_size`` must hold, which it does for
#: the defaults (4 <= 16).
#: Problem knob: XOR's ``error_method`` — the loss is negated into a fitness, so
#: ``"mse"`` with ``fitness_target=-0.01`` means "mean squared error <= 0.01".
#: Pipeline knobs: the documented ``Pipeline.__init__`` arguments.
TINY_CONFIG = {
    # genome.DefaultGenome
    "num_inputs": 2,
    "num_outputs": 1,
    "max_nodes": 8,
    "max_conns": 16,
    "max_in_degree": 4,
    "max_depth": 8,
    "output_transform": "sigmoid",
    # algorithm.NEAT
    "pop_size": 16,
    "species_size": 4,
    # problem.XOR
    "error_method": "mse",
    # Pipeline
    "seed": 0,
    "fitness_target": -0.01,
    "generation_limit": 60,
    "backend": "numpy",
    "device": None,
    "compile_options": None,
    "is_save": False,
    "save_dir": None,
    "is_detailed": True,
    "show_problem_details": False,
}

#: The columns ``Pipeline.setup`` writes into ``<save_dir>/log.txt`` (header first).
LOG_COLUMNS = ("Generation", "Max", "Min", "Mean", "Std", "Cost Time")

#: The ``Pipeline.__init__`` parameters that only the evoxc compile site provides.
PIPELINE_COMPILE_KWARGS = ("backend", "device", "compile_options", "is_detailed")

#: The evoxc entry points that compile a graph; :func:`record_compiles` wraps whichever exist.
COMPILE_ENTRY_POINTS = ("build", "build_cached", "lower", "compile")


# --------------------------------------------------------------------------- #
# structural probe of the loaded pipeline module
# --------------------------------------------------------------------------- #


def _top_level_jax_imports(module: Any) -> list[tuple[int, str]]:
    """Find the unindented ``import jax`` / ``from jax`` statements of a module's source.

    Args:
        module: The loaded module to inspect; its source file is read, never imported.

    Returns:
        ``(line_number, statement)`` pairs for every top-level JAX import, in file
        order. An unreadable or file-less module yields ``[]``: the probe reports
        what it can see instead of guessing.

    Notes:
        Reading ``module.__file__`` is deliberate — importing ``jax`` to test whether
        the pipeline binds it would defeat the import-hygiene gate this suite guards.
    """
    path = getattr(module, "__file__", None)
    if not path:
        return []
    try:
        lines = Path(path).read_text(encoding="utf-8").splitlines()
    except OSError:
        return []
    hits: list[tuple[int, str]] = []
    for number, line in enumerate(lines, start=1):
        if not line or line[:1].isspace():  # blank or indented: not a module-level import
            continue
        statement = line.strip()
        if statement.startswith(("import jax", "from jax")):
            hits.append((number, statement))
    return hits


def missing_evoxc_pipeline_features(tn: Any) -> list[str]:
    """List the evoxc-pipeline features the *loaded* ``tensorneat`` package is missing.

    The check is structural, so it flips without any test edit the moment
    ``src/tensorneat/pipeline.py`` is replaced: it never names a date, a target
    release or an expected failure.

    Args:
        tn: The imported ``tensorneat`` package (the ``tn`` fixture).

    Returns:
        One human-readable string per missing feature, empty when the evoxc compile
        site has landed. The features are:

        * each of :data:`PIPELINE_COMPILE_KWARGS` absent from
          ``inspect.signature(tn.Pipeline.__init__).parameters``;
        * the loaded ``tensorneat.pipeline`` module still binding a ``jax``/``jnp``
          name;
        * a top-level ``import jax`` / ``from jax`` line in that module's source.
    """
    missing: list[str] = []
    pipeline_cls = getattr(tn, "Pipeline", None)
    if pipeline_cls is None:
        return ["tensorneat exports no Pipeline"]

    try:
        parameters = inspect.signature(pipeline_cls.__init__).parameters
    except (TypeError, ValueError) as exc:  # a C-level or missing signature
        missing.append(f"Pipeline.__init__ has no inspectable signature ({exc})")
    else:
        for name in PIPELINE_COMPILE_KWARGS:
            if name not in parameters:
                missing.append(f"Pipeline.__init__ lacks the {name!r} parameter")

    module = inspect.getmodule(pipeline_cls) or sys.modules.get("tensorneat.pipeline")
    if module is None:
        missing.append("tensorneat.pipeline is not importable")
        return missing

    bound = [name for name in ("jax", "jnp") if hasattr(module, name)]
    if bound:
        missing.append(f"{module.__name__} binds {', '.join(bound)}")
    for number, statement in _top_level_jax_imports(module):
        missing.append(f"{module.__name__}:{number} has a top-level `{statement}`")
    return missing


def require_evoxc_pipeline(tn: Any) -> None:
    """Skip the calling test while the pipeline is still the legacy JAX implementation.

    Args:
        tn: The imported ``tensorneat`` package (the ``tn`` fixture).

    Raises:
        pytest.skip.Exception: When :func:`missing_evoxc_pipeline_features` reports
            anything; the reason names every missing feature so a reader knows
            exactly which part of ``pipeline.py`` has not landed yet.

    Notes:
        The whole directory is guarded by the autouse ``_evoxc_pipeline_required``
        fixture, so a test module does not have to call this itself; it stays public
        because a module-scoped fixture that wants the guard established once
        (before it builds anything) reads better than relying on autouse ordering.
    """
    missing = missing_evoxc_pipeline_features(tn)
    if missing:
        pytest.skip(
            "pipeline.py is still the legacy JAX implementation: "
            + "; ".join(missing)
            + "; the evoxc compile site has not landed"
        )


@pytest.fixture(autouse=True)
def _evoxc_pipeline_required(tn):
    """Skip every test of this directory until the evoxc ``Pipeline`` exists.

    This is the self-flipping half of the probe: nothing here is hardcoded, so the
    moment ``missing_evoxc_pipeline_features`` returns an empty list the guard
    becomes a no-op and the suite starts running for real.
    """
    require_evoxc_pipeline(tn)


# --------------------------------------------------------------------------- #
# lazy inputs
# --------------------------------------------------------------------------- #


@pytest.fixture(scope="module")
def tn():
    """The ``tensorneat`` package under test: ``Pipeline``, ``algorithm``, ``genome``, ``problem``, ``common``.

    Returns:
        The imported package, so a test reads ``tn.Pipeline``,
        ``tn.algorithm.NEAT``, ``tn.genome.DefaultGenome``, ``tn.problem.XOR`` and
        ``tn.common.ACT`` exactly as ``src/tensorneat/CONTEXT.md``'s "Public API
        Surface" spells them.

    Notes:
        The import happens at fixture time (module docstring): a
        ``NotImplementedError`` raised during the import becomes a named skip,
        while every other error surfaces. The module is imported once per module
        scope, so a broken import is reported once per test module.
    """
    from support.lazy import require

    return require("tensorneat")


@pytest.fixture(scope="module")
def payloads():
    """The hand-built gene payloads of ``../../support/payloads.py``.

    Returns:
        The ``support.payloads`` module: ``new_payload``, ``set_node``,
        ``set_conn`` and the canonical topology factories, used here to build an
        *off-shape* population (the deliberately-incompatible shape a rebuild
        test feeds to a cached executable instead of letting it reuse one).

    Notes:
        ``support.payloads`` imports ``tensorneat.common.tools`` at module scope,
        so while ``src/`` is unfinished this fixture skips too, naming the payload
        module in the reason.
    """
    from support.lazy import require

    return require("support.payloads")


@pytest.fixture(scope="module")
def runner():
    """The one sanctioned bridge to ``evoxc`` (``../../support/runner.py``).

    Returns:
        The ``support.runner`` module: ``run_defn`` (build + run one call),
        ``to_numpy`` (unwrap ``evoxc.Tensor`` leaves) and ``build_defn``.

    Notes:
        Nothing else in this directory may call ``evoxc.build``/``run``/``lower``/
        ``compile``/``load``; the whole-pipeline tests reach ``evoxc`` through
        ``Pipeline`` itself and the re-scoring helpers reach it here.
    """
    from support.lazy import require

    return require("support.runner")


# --------------------------------------------------------------------------- #
# one tiny pipeline
# --------------------------------------------------------------------------- #


@pytest.fixture(scope="module")
def tiny_pipeline(tn):
    """Factory for a configured-but-not-run ``Pipeline`` (:data:`TINY_CONFIG`, overridable).

    Returns:
        ``make(**overrides) -> Pipeline``: builds ``DefaultGenome`` → ``NEAT`` →
        ``XOR`` → ``Pipeline`` from :data:`TINY_CONFIG` with ``overrides`` applied
        to the knobs of the same name. Nothing is run: a test calls
        ``pipeline.setup()`` itself (or goes through :func:`run_pipeline`).

    Raises:
        TypeError: If ``overrides`` names a knob :data:`TINY_CONFIG` does not
            have. A silently ignored override would make a test assert a
            configuration it never built, so this is a hard error.

    Notes:
        ``output_transform`` is given as a registry *name* (``"sigmoid"``) and
        resolved against ``tn.common.ACT``; a callable or ``None`` passes through
        unchanged. That is what keeps the XOR read-out inside ``(0, 1)``, exactly
        like the shipped ``examples/func_fit/xor.py`` — without it the ``mse <=
        0.01`` target is not reachable for a population this small.
        The pipeline is built with the documented evoxc arguments; while the
        legacy class is still loaded the constructor rejects them, which is fine
        because :func:`require_evoxc_pipeline` skips every test before it gets here.
    """

    def make(**overrides):
        unknown = sorted(set(overrides) - set(TINY_CONFIG))
        if unknown:
            raise TypeError(
                f"unknown tiny-pipeline knob(s) {unknown}; "
                f"known knobs are {sorted(TINY_CONFIG)}"
            )
        config = dict(TINY_CONFIG)
        config.update(overrides)

        output_transform = config["output_transform"]
        if isinstance(output_transform, str):
            output_transform = getattr(tn.common.ACT, output_transform)

        genome = tn.genome.DefaultGenome(
            num_inputs=config["num_inputs"],
            num_outputs=config["num_outputs"],
            max_nodes=config["max_nodes"],
            max_conns=config["max_conns"],
            max_in_degree=config["max_in_degree"],
            max_depth=config["max_depth"],
            output_transform=output_transform,
        )
        algorithm = tn.algorithm.NEAT(
            genome,
            pop_size=config["pop_size"],
            species_size=config["species_size"],
        )
        problem = tn.problem.XOR(error_method=config["error_method"])
        return tn.Pipeline(
            algorithm,
            problem,
            seed=config["seed"],
            fitness_target=config["fitness_target"],
            generation_limit=config["generation_limit"],
            backend=config["backend"],
            device=config["device"],
            compile_options=config["compile_options"],
            is_save=config["is_save"],
            save_dir=config["save_dir"],
            is_detailed=config["is_detailed"],
            show_problem_details=config["show_problem_details"],
        )

    return make


@pytest.fixture(scope="module")
def run_pipeline():
    """Run a whole ``Pipeline`` to its stopping condition and return what it produced.

    Returns:
        ``run(pipeline) -> (state, best)``: exactly what
        ``pipeline.auto_run(pipeline.setup())`` returns — the final ``State`` and
        the reported best genome. The per-generation trajectory is deliberately
        *not* part of this pair: it comes from :func:`step_loop`, which drives the
        documented ``pipeline.step`` instead of an internal history attribute.

    Notes:
        ``setup()`` is called per run, on a *fresh* ``Pipeline``: the two
        determinism runs must not share an initialisation key, and a run must be
        reproducible from ``(seed, config, shapes)`` alone. The helper stops being
        meaningful for a pipeline that was already stepped — ``auto_run`` starts
        from the state it is handed, not from generation 0.
    """

    def run(pipeline):
        state = pipeline.setup()
        state, best = pipeline.auto_run(state)
        return state, best

    return run


# --------------------------------------------------------------------------- #
# generation driving, without an undocumented history attribute
# --------------------------------------------------------------------------- #


def step_once(pipeline: Any, state: Any) -> tuple[Any, np.ndarray]:
    """Advance a pipeline by exactly one generation and return its fitness on the host.

    Args:
        pipeline: The ``Pipeline`` under test; only its documented ``step`` method
            and ``pop_size`` attribute are read.
        state: The current ``State`` (the one ``setup``/``auto_run``/a previous
            ``step`` returned).

    Returns:
        ``(state, fitness)``: the state returned by ``pipeline.step`` and a numpy
        ``(pop_size,)`` float32/float64 fitness vector, unwrapped from whatever
        tensor type the backend produced.

    Raises:
        AssertionError: If ``pipeline.step`` returns anything but the documented
            ``(state, ..., fitness)`` tuple/list — fewer than two elements, a
            fitness that is not one-dimensional, a length that disagrees with
            ``pipeline.pop_size``, or a non-floating dtype. The message quotes the
            contract instead of guessing what the caller meant.

    Notes:
        The benchmark adapter (``benchmarks/harness/adapters/new.py``) accepts both
        ``(state, pop, fitness)`` and ``(state, fitness)``, so this helper reads
        ``result[0]`` and ``result[-1]`` and validates the rest, which pins the same
        contract from the test side.
    """
    from support.runner import to_numpy  # deferred: support/ is not importable at collection

    result = pipeline.step(state)
    if not isinstance(result, (tuple, list)):
        raise AssertionError(
            "pipeline.step must return the documented (state, ..., fitness) "
            f"tuple/list, got {type(result).__name__}"
        )
    if len(result) < 2:
        raise AssertionError(
            "pipeline.step must return at least (state, fitness); got a "
            f"{len(result)}-element sequence"
        )

    new_state = result[0]
    fitness = np.asarray(to_numpy(result[-1]))
    if fitness.ndim != 1:
        raise AssertionError(
            "pipeline.step must return a (pop_size,) fitness array, got shape "
            f"{fitness.shape}"
        )
    pop_size = getattr(pipeline, "pop_size", None)
    if pop_size is not None and fitness.shape[0] != int(pop_size):
        raise AssertionError(
            f"pipeline.step returned {fitness.shape[0]} fitness value(s) for a "
            f"pipeline whose pop_size is {int(pop_size)}"
        )
    if not np.issubdtype(fitness.dtype, np.floating):
        raise AssertionError(
            f"pipeline.step fitness must be float32/float64, got {fitness.dtype}"
        )
    return new_state, fitness


def step_loop(
    pipeline: Any,
    state: Any,
    generations: int | None = None,
    target: float | None = None,
) -> tuple[Any, list[float]]:
    """Drive ``pipeline.step`` for a fixed number of generations, stopping early on a target.

    Args:
        pipeline: The ``Pipeline`` under test.
        state: The state ``setup()`` returned.
        generations: How many generations to run at most; ``None`` means
            ``pipeline.generation_limit``.
        target: Stop as soon as a generation's best fitness is ``>= target``;
            ``None`` means "never stop early", which is what a fixed-length
            determinism run wants.

    Returns:
        ``(state, history)``: the state of the last generation and the per
        generation best (maximum) fitness as Python floats, in generation order.
        ``len(history)`` is therefore the number of generations actually run.

    Raises:
        ValueError: If ``generations`` is negative.
        AssertionError: Propagated from :func:`step_once` when a step violates the
            documented return contract.

    Notes:
        The recorded number is the raw maximum of the fitness vector, the same
        comparison ``auto_run`` uses to decide whether the target was reached; a
        generation whose whole population is invalid records ``-inf``. NaN cannot
        reach here by contract (every problem maps NaN fitness to ``-inf``).
    """
    if generations is None:
        generations = int(getattr(pipeline, "generation_limit", 1))
    if generations < 0:
        raise ValueError(f"generations must be >= 0, got {generations}")

    history: list[float] = []
    for _ in range(int(generations)):
        state, fitness = step_once(pipeline, state)
        best = float(np.max(fitness))
        history.append(best)
        if target is not None and best >= target:
            break
    return state, history


# --------------------------------------------------------------------------- #
# compile-call instrumentation
# --------------------------------------------------------------------------- #


@contextlib.contextmanager
def record_compiles() -> Iterator[list[tuple[str, dict[str, Any]]]]:
    """Record every call into the evoxc compile entry points made inside the block.

    Yields:
        A list that grows one ``(entry_point_name, keyword_arguments)`` pair per
        call while the block runs, e.g. ``[("build", {"backend": "numpy"})]``.

    Notes:
        ``src/`` reaches the compiler exclusively through ``evoxc.build`` (see
        ``algorithm/neat/neat.py``: ``evoxc.build(_init_kernel, spec, **options)``),
        so wrapping that module attribute is a legitimate observation point instead
        of a guess about an internal cache: a test can prove "the second generation
        of the same shape does not compile again" without naming any cache attribute.
        Only the entry points that exist are wrapped (``evoxc.build_cached`` does
        not exist today, and ``hasattr`` decides). Names already imported into an
        ``tensorneat`` module (``from evoxc import build``) are rebound to the same
        wrapper, so the count does not depend on the import style.
        Every original is restored in a ``finally``; a failing block never leaves a
        counting wrapper installed.
    """
    import evoxc  # deferred: the module is a hard dependency, but collection needs nothing

    calls: list[tuple[str, dict[str, Any]]] = []
    patched: list[tuple[Any, str, Any]] = []

    def wrap(name: str, original: Callable[..., Any]) -> Callable[..., Any]:
        def recorded(*args: Any, **kwargs: Any) -> Any:
            calls.append((name, dict(kwargs)))
            return original(*args, **kwargs)

        recorded.__name__ = f"recorded_{name}"
        recorded.__wrapped__ = original  # type: ignore[attr-defined]
        return recorded

    try:
        for name in COMPILE_ENTRY_POINTS:
            original = getattr(evoxc, name, None)
            if original is None:
                continue
            wrapper = wrap(name, original)
            setattr(evoxc, name, wrapper)
            patched.append((evoxc, name, original))
            for module_name, module in list(sys.modules.items()):
                if module is None or not (
                    module_name == "tensorneat" or module_name.startswith("tensorneat.")
                ):
                    continue
                for attribute, value in list(vars(module).items()):
                    if value is original:  # an alias imported before the block opened
                        setattr(module, attribute, wrapper)
                        patched.append((module, attribute, original))
        yield calls
    finally:
        for module, attribute, original in reversed(patched):
            setattr(module, attribute, original)


@pytest.fixture
def compiles():
    """Factory returning the :func:`record_compiles` context manager.

    Returns:
        ``compiles()`` used as ``with compiles() as calls:`` — the recorded
        ``(entry_point, kwargs)`` list grows while the block runs.
    """
    return record_compiles


# --------------------------------------------------------------------------- #
# host-side re-scoring through the documented kernel triple
# --------------------------------------------------------------------------- #


def _payload_arrays(pop: Any) -> tuple[np.ndarray, ...]:
    """Split a gene payload into its arrays, whatever container it arrived in.

    Args:
        pop: A ``genome.utils.GenomeTensors`` (a 4-field namedtuple), a plain
            tuple/list of the four payload arrays, or a ``support.payloads.Payload``
            (which exposes the same arrays as ``.args``).

    Returns:
        The four payload arrays in documented order.

    Raises:
        TypeError: If ``pop`` is none of the accepted containers. Guessing an
            order would silently score the wrong arrays.
    """
    if getattr(pop, "_fields", None) is not None:  # GenomeTensors and friends
        return tuple(pop)
    args = getattr(pop, "args", None)  # support.payloads.Payload
    if args is not None:
        return tuple(args)
    if isinstance(pop, (tuple, list)) and len(pop) == 4:
        return tuple(pop)
    raise TypeError(
        "evaluate_population/transform_of take a GenomeTensors payload, a 4-array "
        f"tuple/list or a support.payloads.Payload; got {type(pop).__name__}"
    )


def _population_size(payload: Any) -> int:
    """Read the static population size off a payload's leading axis.

    Args:
        payload: The rebuilt ``GenomeTensors`` inside the active trace (every leaf
            carries the population on axis 0).

    Returns:
        ``P`` as a Python int; static shapes make this host-visible even mid-trace.
    """
    return int(tuple(payload)[0].shape[0])


def _repeat_along_pop(field: Any, reps: int, pop_size: int) -> Any:
    """Repeat a batch-first ``(P, ...)`` tensor to ``(P * reps, ...)``, individual-major.

    Args:
        field: A traced tensor whose axis 0 is the individual.
        reps: How many copies of every individual are needed (``S``, the number of
            problem samples, when the schedule is tiled for a func-fit problem).
        pop_size: The static ``P`` of ``field``.

    Returns:
        ``(P * reps, ...)`` with rows ``p * reps + s`` holding individual ``p``.

    Notes:
        This is the exact order ``problem.FuncFit`` tiles its dataset in, so row
        ``p * S + s`` of the returned tensor is sample ``s`` of individual ``p``.
        The recipe is ``reshape`` to ``(P, 1, ...)`` → ``evoxc.broadcast`` to
        ``(P, reps, ...)`` → ``reshape`` to ``(P * reps, ...)``. ``evoxc`` has no
        ``broadcast_to`` name (``evoxc.broadcast`` is the numpy ``broadcast_to``
        semantics), and writing ``reps`` into the middle axis of the target shape is
        what makes the repetition individual-major.
    """
    import evoxc  # deferred: only needed while a graph is being traced

    if reps == 1:
        return field
    tail = tuple(int(size) for size in field.shape[1:])
    expanded = evoxc.reshape(field, (pop_size, 1) + tail)
    widened = evoxc.broadcast(expanded, (pop_size, reps) + tail)
    return evoxc.reshape(widened, (pop_size * reps,) + tail)


def _replicate(container: Any, reps: int, pop_size: int) -> Any:
    """Replicate every leaf of a batch-first pytree along the population axis.

    Args:
        container: A ``Transformed`` schedule or a ``GenomeTensors`` payload (any
            namedtuple/tuple whose leaves are ``(P, ...)`` tensors).
        reps: Copies per individual.
        pop_size: The static ``P`` of every leaf.

    Returns:
        The same container type holding ``(P * reps, ...)`` leaves.
    """
    if reps == 1:
        return container
    return type(container)(
        *(_repeat_along_pop(field, reps, pop_size) for field in container)
    )


@pytest.fixture(scope="module")
def evaluate_population(runner):
    """Score a population payload on the host, through the documented kernel contract.

    Returns:
        ``evaluate(pipeline, state, pop) -> (P,) numpy fitness``: derives the
        execution schedule with ``pipeline.algorithm.transform(state, pop)``, hands
        ``pipeline.problem.evaluate`` an ``act_fn`` that replicates both the
        schedule and the payload to the problem's tiled batch, and runs the whole
        thing as one graph through :mod:`support.runner`. It accepts a payload with
        ``P = 1`` (the "re-score the reported best genome" case) and one with
        ``P = pipeline.pop_size``.

    Notes:
        ``problem.evaluate`` (func-fit) tiles the dataset to ``(P * S, I)`` and calls
        ``act_fn`` **once** with that batch, row ``p * S + s`` being sample ``s`` of
        individual ``p``; ``algorithm.forward`` still expects one schedule leaf per
        row, so ``act_fn`` repeats the schedule and the payload ``S = rows // P``
        times with :func:`_replicate`. Without that repetition the genome kernel
        raises a ``ShapeError`` on mismatched concatenation.
        ``pop`` may be a ``GenomeTensors``, a 4-array tuple/list or a
        ``support.payloads.Payload``; it is rebuilt as ``type(state.pop)`` inside the
        trace so the algorithm receives its own payload container. ``state.randkey``
        is the single key the problem protocol takes — the XOR problem never
        consumes it. The result comes back through ``support.runner.to_numpy``, so a
        test compares plain numpy against the run's reported fitness.
    """

    def evaluate(pipeline, state, pop):
        algorithm = pipeline.algorithm
        problem = pipeline.problem
        payload_type = type(state.pop)
        leaves = _payload_arrays(pop)
        pop_size = int(leaves[0].shape[0])

        def traced(*arrays):
            payload = payload_type(*arrays)
            transformed = algorithm.transform(state, payload)

            def act_fn(batch):
                rows = int(batch.shape[0])
                if rows % pop_size:
                    raise AssertionError(
                        f"the problem tiled {rows} rows, which is not a multiple of "
                        f"the {pop_size}-individual schedule; check that the payload "
                        "and the schedule describe the same population"
                    )
                reps = rows // pop_size
                schedule = _replicate(transformed, reps, pop_size)
                replicated = _replicate(payload, reps, pop_size)
                return algorithm.forward(state, schedule, batch, pop=replicated)

            return problem.evaluate(state, state.randkey, act_fn, transformed)

        fitness = np.asarray(runner.run_defn(traced, *leaves))
        if fitness.shape != (pop_size,):
            raise AssertionError(
                f"problem.evaluate must return ({pop_size},) fitness for this "
                f"payload, got {fitness.shape}"
            )
        return fitness

    return evaluate


def transform_of(pipeline: Any, state: Any, pop: Any) -> Any:
    """Return the host-visible execution schedule of a payload.

    Args:
        pipeline: The ``Pipeline`` under test (its ``algorithm`` derives the schedule).
        state: The ``State`` the payload belongs to.
        pop: A ``GenomeTensors`` payload, a 4-array tuple/list or a
            ``support.payloads.Payload`` — same containers as
            :func:`evaluate_population`.

    Returns:
        The genome's ``Transformed`` namedtuple with every ``evoxc.Tensor`` leaf
        replaced by its numpy view, so a test can assert on ``edge_overflow``,
        ``node_exists`` or ``in_valid`` without touching a trace.

    Notes:
        This exists for the static-shape tests: ``edge_overflow`` is how a
        schedule reports that the configured ``max_in_degree`` was too small, and
        reading it from the host beats reaching into ``evoxc`` from the test.
    """
    from support.runner import run_defn  # deferred: support/ is not importable at collection

    algorithm = pipeline.algorithm
    payload_type = type(state.pop)
    leaves = _payload_arrays(pop)

    def traced(*arrays):
        return algorithm.transform(state, payload_type(*arrays))

    return run_defn(traced, *leaves)


# --------------------------------------------------------------------------- #
# optional trajectory read-back for the saved-log tests
# --------------------------------------------------------------------------- #


def fitness_history(pipeline: Any) -> list[tuple[int, float]]:
    """Read back the per-generation ``(generation, max_fitness)`` of a finished run.

    Args:
        pipeline: A ``Pipeline`` whose ``auto_run`` has returned (or one with
            ``is_save=True`` that has written at least one log line).

    Returns:
        A list of ``(generation, max_fitness)`` pairs in generation order, the
        maximum taken over the generation's *valid* fitness values — the same
        number ``Pipeline`` prints as ``max`` and writes into the ``Max`` column.

    Raises:
        RuntimeError: If the pipeline exposes neither ``fitness_history`` nor a
            readable ``<save_dir>/log.txt``. Returning an empty list would make
            "nothing was recorded" indistinguishable from "the run was
            deterministic and empty", so this fails loudly and names both
            candidate contracts.

    Notes:
        This is only for the ``is_save=True`` log assertions; a trajectory a test
        wants to compare across runs comes from :func:`step_loop`, which reads the
        documented ``pipeline.step`` instead of an undocumented attribute.
        ``fitness_history``, when present, wins over the log: it is the in-memory
        record of the same numbers and does not require ``is_save``. A ``NaN``
        ``Max`` cell (a generation with no valid fitness at all) parses to
        ``float("nan")`` rather than raising, so a run that briefly produced no
        evaluable genome still yields the full-length sequence.
    """
    recorded = getattr(pipeline, "fitness_history", None)
    if recorded is not None:
        return [(int(generation), float(best)) for generation, best in recorded]

    save_dir = getattr(pipeline, "save_dir", None)
    if getattr(pipeline, "is_save", False) and save_dir is not None:
        rows = _read_log(save_dir)
        if rows:
            return rows

    raise RuntimeError(
        "no per-generation fitness trajectory: the pipeline exposes neither "
        "`pipeline.fitness_history` nor a readable `<save_dir>/log.txt` "
        f"(is_save={getattr(pipeline, 'is_save', None)!r}, save_dir={save_dir!r})"
    )


def _read_log(save_dir: Any) -> list[tuple[int, float]]:
    """Parse ``<save_dir>/log.txt`` into ``(generation, max_fitness)`` pairs.

    Args:
        save_dir: The directory ``Pipeline.setup`` wrote ``log.txt`` into (a
            ``str``/``Path``, or the ``tmp_path`` fixture's ``Path``).

    Returns:
        The parsed pairs in file order; an absent or header-only log yields ``[]``
        so the caller can decide whether that is an error.

    Raises:
        ValueError: If a line does not have exactly :data:`LOG_COLUMNS` cells.
    """
    path = Path(save_dir) / "log.txt"
    if not path.is_file():
        return []
    rows = []
    for line in path.read_text().splitlines()[1:]:  # line 0 is the header
        cells = line.split(",")
        if len(cells) != len(LOG_COLUMNS):
            raise ValueError(
                f"{path}: expected {len(LOG_COLUMNS)} comma-separated columns "
                f"{LOG_COLUMNS}, got {len(cells)} in {line!r}"
            )
        rows.append((int(cells[0]), float(cells[1])))
    return rows
