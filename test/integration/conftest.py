"""Fixtures shared by the ``./integration`` suite: one configured ``Pipeline`` and three helpers.

What lives here, and why
------------------------
Every test in this directory is a *whole-pipeline* test, so the thing under test
is a ``tensorneat.Pipeline`` and not a kernel. That makes four problems worth
solving once, here:

**1. Nothing may be imported at module scope.** ``import tensorneat`` still
raises ``NotImplementedError`` (``tensorneat/algorithm/base.py`` evaluates
``State()`` as a default argument while the class body is created, and ``State``
is a design skeleton), so a module-scope ``from tensorneat import Pipeline`` would
turn every module here into a collection error instead of a named skip. The three
lazily resolved fixtures (:func:`tn`, :func:`payloads`, :func:`runner`) go through
:func:`support.lazy.require`, which converts *only* that pending-implementation
error into a skip and lets an ``ImportError`` surface. ``import support.lazy`` is
deferred for the same reason ``../conftest.py`` prepends ``test/`` to
``sys.path``: it must happen after collection, not while this file is imported.

**2. One tiny-but-real pipeline configuration.** ``Pipeline(algorithm,
problem, seed, fitness_target, generation_limit, backend)`` takes a whole
``NEAT(DefaultGenome(...), pop_size=...)`` instance and a ``XOR()`` problem, so a
test that wants "the same run, but at ``pop_size=8``" would otherwise re-spell a
dozen genome knobs. :func:`tiny_pipeline` owns that spelling, keys each knob to
its owner object (genome / NEAT / problem / pipeline), and refuses an unknown
knob instead of silently ignoring it.

**3. The generation loop, not its internals.** :func:`run_pipeline` is
``setup()`` + ``auto_run()`` + a history read-back, so a test asserts on what
``auto_run`` *returns* rather than on the pipeline's loop.

**4. Re-scoring one genome on the host.** ``Pipeline`` reports a best genome but
has no "score this genome" entry point; :func:`evaluate_population` closes that
gap with the documented kernel contract (``algorithm.transform`` →
``algorithm.forward`` → ``problem.evaluate``) through ``support.runner``, which is
the only module in the suite allowed to call ``evoxc.build``/``run``.

The one underspecified point (reported to the parent)
----------------------------------------------------
``src/tensorneat/CONTEXT.md`` documents ``Pipeline(...)``'s constructor arguments
and its four methods, but not how a caller reads the per-generation fitness
trajectory back. :func:`fitness_history` therefore accepts the two shapes the
documented behaviour can take — an in-memory ``pipeline.fitness_history``
sequence of ``(generation, max_fitness)`` pairs, or the ``log.txt`` that
``is_save=True`` writes into ``save_dir`` — and raises a ``RuntimeError`` naming
both when neither exists. It never returns an empty list, because "no history"
must not look like "a run that logged nothing".

No ``jax``, no ``torch``, no network, no GPU: the backend is ``"numpy"``
everywhere, and nothing here touches wall-clock time.
"""

from __future__ import annotations

from functools import partial
from pathlib import Path

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


@pytest.fixture(scope="module")
def tn():
    """The ``tensorneat`` package under test: ``Pipeline``, ``algorithm``, ``genome``, ``problem``, ``common``.

    Returns:
        The imported package, so a test reads ``tn.Pipeline``,
        ``tn.algorithm.NEAT``, ``tn.genome.DefaultGenome``, ``tn.problem.XOR`` and
        ``tn.common.ACT`` exactly as ``src/tensorneat/CONTEXT.md``'s "Public API
        Surface" spells them.

    Notes:
        The import is deferred (module docstring): a collection-time import would
        raise ``NotImplementedError`` and turn every module in this directory into
        an error instead of a named skip. The module is imported once per module
        scope, so a broken ``src/`` is reported once per test module.
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
        ``Pipeline`` itself and the re-scoring helper reaches it here.
    """
    from support.lazy import require

    return require("support.runner")


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
        ``run(pipeline) -> (state, best, history)``: ``state`` and ``best`` are
        exactly what ``pipeline.auto_run(pipeline.setup())`` returns (the final
        ``State`` and the reported best genome), and ``history`` is the
        per-generation ``(generation, max_fitness)`` list from
        :func:`fitness_history`.

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
        return state, best, fitness_history(pipeline)

    return run


@pytest.fixture(scope="module")
def evaluate_population(runner):
    """Score a population payload on the host, through the documented kernel contract.

    Returns:
        ``evaluate(pipeline, state, pop) -> (P,) numpy fitness``: derives the
        execution schedule with ``pipeline.algorithm.transform(state, pop)``,
        partial-applies ``algorithm.forward`` into the ``act_fn`` the problem
        protocol expects, and calls
        ``pipeline.problem.evaluate(state, state.randkey, act_fn, transformed)``.

    Notes:
        This is how a test checks a *reported* best genome: the value has to come
        out of the same three kernels the generation loop used, so a stale
        ``best_genome`` shows up as a fitness mismatch instead of passing on the
        logged number alone. ``pop`` is a genome payload pytree (``GenomeTensors``
        order); the traced closure rebuilds it from its leaves with ``_make`` when
        it is a namedtuple and as a plain tuple otherwise, so the helper does not
        pin the payload's arity. ``state.randkey`` is the single key the problem
        protocol takes — the XOR problem never consumes it.
    """

    def evaluate(pipeline, state, pop):
        algorithm = pipeline.algorithm
        problem = pipeline.problem
        leaves = tuple(pop)
        make = getattr(pop, "_make", None)

        def rebuild(*arrays):
            return make(*arrays) if make is not None else tuple(arrays)

        def traced(*arrays):
            pop_in = rebuild(*arrays)
            transformed = algorithm.transform(state, pop_in)
            act_fn = partial(algorithm.forward, state, transformed)
            return problem.evaluate(state, state.randkey, act_fn, transformed)

        return runner.run_defn(traced, *leaves)

    return evaluate


def fitness_history(pipeline):
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
            candidate contracts (see the module docstring: the trajectory
            accessor is the one part of the documented ``Pipeline`` contract that
            is underspecified, and it is reported to the parent).

    Notes:
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


def _read_log(save_dir):
    """Parse ``<save_dir>/log.txt`` into ``(generation, max_fitness)`` pairs.

    Args:
        save_dir: The directory ``Pipeline.setup`` wrote ``log.txt`` into (a
            ``str``/``Path``, or the ``tmp_path`` fixture's ``Path``).

    Returns:
        The parsed pairs in file order; an absent or header-only log yields ``[]``
        so the caller can decide whether that is an error.
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
