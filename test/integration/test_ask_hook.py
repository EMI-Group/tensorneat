"""The ``ask`` hook: one call per build, and its result is what gets evaluated (marker ``slow``).

What this module proves
-----------------------
``Pipeline``'s generation kernel calls ``algorithm.ask(current)`` once, *inside the
traced graph*, and hands the result to ``algorithm.transform``/``problem.evaluate``
(``src/tensorneat/pipeline.py``:591). Three properties follow, and each is asserted
here against a subclass of ``NEAT`` that does nothing but observe the hook, so the
pipeline is exercised through its documented surface only:

* **``ask`` runs when the graph is built, once per build.** It is *not* "once per
  generation": a shape signature compiles an executable that already contains the
  call, and every later generation of the same shape reuses it. So the invariant is
  ``ask_calls == pipeline.builds`` at every point of a run -- ``0`` after
  ``setup()`` (host-only, builds nothing), ``1`` after the first ``step`` (which
  builds), still ``1`` after a second step of the same shape. A pipeline that asked
  per generation would show ``ask_calls > builds``; a pipeline that never asked
  would show ``0`` while stepping.
* **The returned population really is the evaluated one.** A second subclass zeroes
  the payload's two *attribute* arrays (keeping the keys, so the topology and every
  static shape are unchanged) inside ``ask``. With ``XOR``'s negated ``mse`` and an
  all-zero read-out, ``output_transform="sigmoid"`` maps every node value through
  ``sigmoid(0) = 0.5``, and the four targets ``[0, 1, 1, 0]`` give
  ``-(0.25 + 0.25 + 0.25 + 0.25) / 4 = -0.25`` exactly for every individual. The
  control is the unmodified hook at the same seed: its fitness is *not* ``-0.25``,
  so the value is evidence about ``ask`` and not about ``XOR``'s arithmetic.
* **The documented reporting mismatch stays a mismatch.** ``tell``, the returned
  ``previous_pop`` and the checkpoints describe ``state.pop``, never the ``ask``
  result (``src/tensorneat/pipeline.py``:563-567). With the zeroing subclass in
  place, the fitness is the all-zero read-out (``-0.25``) while ``step``'s returned
  ``previous_pop`` is still the pre-step ``state.pop``, array for array.

The oracles
-----------
The subclass counters themselves, ``Pipeline.builds`` (a documented host-visible
attribute), the ``(P,)`` fitness ``Pipeline.step`` returns, and the pre-step
``state.pop`` the test read *before* stepping. No recorded number is compared: the
one closed-form value, ``-0.25``, is derived above from ``XOR``'s targets and the
sigmoid read-out, and asserted with ``atol=1e-6``.

Budgets (measured, never asserted)
----------------------------------
Every pipeline is ``pop_size=8`` or ``16``, ``max_nodes=8``, ``max_conns=16`` and
``generation_limit`` 1 or 2 with ``fitness_target=1.0`` (unreachable: ``XOR``
fitness is ``-mse <= 0``), so every run consumes exactly its budget and compiles
once. ``is_detailed=False`` keeps the run silent; nothing here reads a clock.

Why this file has no ``tensorneat`` import at module scope
---------------------------------------------------------
The package arrives through the ``tn`` fixture, which resolves it with
``support.lazy.require`` at fixture time; every subclass and every ``evoxc`` import
is created inside a fixture or a test. The directory-wide autouse probe
(:func:`~.conftest.require_evoxc_pipeline`) is the only guard and it fails rather
than skips on this tree.
"""

from __future__ import annotations

import numpy as np
import pytest

from .conftest import TINY_CONFIG

pytestmark = [pytest.mark.slow]

#: The genome knobs every pipeline in this module shares, taken from the suite's
#: one configuration so a change there cannot silently drift from here.
#:
#: ``num_inputs=2``/``num_outputs=1`` is ``XOR``'s per-sample arity,
#: ``max_nodes``/``max_conns`` are the padded gene geometry and
#: ``max_in_degree``/``max_depth`` the static schedule bounds.
GENOME_KNOBS = {
    name: TINY_CONFIG[name]
    for name in (
        "num_inputs",
        "num_outputs",
        "max_nodes",
        "max_conns",
        "max_in_degree",
        "max_depth",
    )
}

#: The population of every narrow pipeline here; small enough to keep the module
#: at a handful of compiles.
POP_SIZE = 8

#: A second, wider population: a different shape signature, so it is its own build.
WIDER_POP_SIZE = TINY_CONFIG["pop_size"]  # 16

#: ``species_size * min_species_size <= pop_size`` must hold for both widths;
#: ``2 * 1 <= 8`` and ``2 * 1 <= 16`` do.
SPECIES_SIZE = 2

#: One seed for every pipeline, so two algorithms see the *same* initial population
#: and their fitness vectors are comparable.
SEED = 0

#: Unreachable for ``XOR`` (fitness is ``-mse <= 0``): every run consumes exactly
#: ``generation_limit`` generations and no early stop can mask a missing build.
UNREACHABLE_TARGET = 1.0

#: The four ``genome.utils.GenomeTensors`` field names, in payload order.
GENOME_FIELDS = ("node_keys", "node_attrs", "conn_keys", "conn_attrs")

#: ``XOR`` fitness of an all-zero read-out: six ``sigmoid(0) = 0.5`` predictions
#: against targets ``[0, 1, 1, 0]`` is a mean squared error of exactly ``0.25``,
#: which the problem negates into ``-0.25``.
ZEROED_READOUT_FITNESS = -0.25


def _ask_algorithm_class(tn, *, zero_attributes: bool = False):
    """Build a ``NEAT`` subclass that counts its ``ask`` calls and may zero the attributes.

    Args:
        tn: The imported ``tensorneat`` package (the ``tn`` fixture). Its
            ``algorithm.NEAT`` is the base class, so ``ask`` keeps the shipped
            behaviour and only the observation is added.
        zero_attributes: When ``True``, ``ask`` returns a payload with the same
            keys but **all-zero** node/connection attribute arrays, which is what
            turns ``XOR``'s fitness into :data:`ZEROED_READOUT_FITNESS`. When
            ``False``, the shipped ``state.pop`` is returned unchanged.

    Returns:
        The subclass. Each instance starts from ``ask_calls = 0`` (an instance
        attribute, never class-level, so two pipelines cannot share a count).

    Notes:
        ``ask`` runs inside an active trace, so the zeroing payload is rebuilt with
        ``evoxc.numpy.zeros`` over the *symbolic* shapes the payload already
        carries; the imports live inside the method because this module may not
        touch ``evoxc`` at module scope either. ``super().ask(state)`` is called
        first so the base hook stays the single source of the payload layout.
    """

    class ProbingNEAT(tn.algorithm.NEAT):
        """``NEAT`` with an instrumented ``ask``; behaviour otherwise unchanged."""

        def __init__(self, *args, **kwargs) -> None:
            super().__init__(*args, **kwargs)
            #: How many times the traced graph has called :meth:`ask` on *this*
            #: algorithm. One per build, not one per generation.
            self.ask_calls = 0

        def ask(self, state):
            """Count the call, then return the population (optionally attribute-zeroed)."""
            self.ask_calls += 1
            pop = super().ask(state)
            if not zero_attributes:
                return pop
            import evoxc
            import evoxc.numpy as enp

            return tn.genome.GenomeTensors(
                pop.node_keys,
                enp.zeros(tuple(pop.node_attrs.shape), dtype=evoxc.float32),
                pop.conn_keys,
                enp.zeros(tuple(pop.conn_attrs.shape), dtype=evoxc.float32),
            )

    return ProbingNEAT


@pytest.fixture(scope="module")
def ask_pipeline(tn):
    """Factory for a small ``XOR`` ``Pipeline`` whose algorithm is supplied by the test.

    Returns:
        ``make(algorithm_class, *, pop_size=POP_SIZE, generation_limit=2) ->
        Pipeline``: ``DefaultGenome`` (:data:`GENOME_KNOBS`, sigmoid read-out) →
        ``algorithm_class`` → ``XOR(error_method="mse")`` → ``Pipeline`` with
        :data:`SEED`, :data:`UNREACHABLE_TARGET` and the numpy backend. Nothing is
        run: the caller calls ``setup()`` and drives ``step``/``auto_run`` itself.

    Notes:
        The algorithm class is a parameter instead of the ``tiny_pipeline`` factory
        because the subclass has to be the *instance* the pipeline holds; every
        other knob is the shared configuration.
    """

    def make(algorithm_class, *, pop_size=POP_SIZE, generation_limit=2):
        genome = tn.genome.DefaultGenome(
            **GENOME_KNOBS, output_transform=tn.common.ACT.sigmoid
        )
        algorithm = algorithm_class(genome, pop_size=pop_size, species_size=SPECIES_SIZE)
        problem = tn.problem.XOR(error_method="mse")
        return tn.Pipeline(
            algorithm,
            problem,
            seed=SEED,
            fitness_target=UNREACHABLE_TARGET,
            generation_limit=generation_limit,
            backend="numpy",
            device=None,
            compile_options=None,
            is_save=False,
            save_dir=None,
            is_detailed=False,
            show_problem_details=False,
        )

    return make


def _generation(tn, state) -> int:
    """Read ``state.generation`` (a 0-d float32 evoxc tensor) as a Python ``int``.

    Args:
        tn: The imported ``tensorneat`` package, for its documented host-side
            accessor ``common.tools.to_numpy`` (an ``evoxc.Tensor`` has no
            ``__array__``, so ``int(...)`` on it raises).
        state: Any ``State`` the pipeline returned.

    Returns:
        The generation counter.
    """
    return int(tn.common.tools.to_numpy(state.generation))


def _payload_arrays(tn, payload) -> tuple[np.ndarray, ...]:
    """Materialise a four-array gene payload on the host, in payload order.

    Args:
        tn: The imported ``tensorneat`` package (its ``common.tools.to_numpy``).
        payload: A ``genome.utils.GenomeTensors`` or any four-element sequence of
            ``evoxc.Tensor``/numpy leaves.

    Returns:
        The four arrays as plain ``numpy`` arrays, ``node_keys``/``node_attrs``/
        ``conn_keys``/``conn_attrs`` in that order.
    """
    return tuple(np.asarray(tn.common.tools.to_numpy(field)) for field in payload)


def _assert_payload_equal(name: str, expected: np.ndarray, actual: np.ndarray) -> None:
    """Assert two payload arrays are equal, exactly, with NaN padding treated as equal.

    Args:
        name: The field name, used in the failure message.
        expected: The array read before the step.
        actual: The array the pipeline returned.

    Raises:
        AssertionError: If the shapes disagree, if an integer array differs at any
            element, or if a floating array differs anywhere other than a NaN
            position that is NaN on both sides.
    """
    assert actual.shape == expected.shape, (
        f"{name}: shape changed across the step, {expected.shape} -> {actual.shape}"
    )
    if np.issubdtype(expected.dtype, np.integer):
        assert np.array_equal(expected, actual), (
            f"{name}: the payload the pipeline reported differs from the pre-step "
            f"state.pop at {int(np.argmax(expected.ravel() != actual.ravel()))}"
        )
    else:
        assert np.array_equal(expected, actual, equal_nan=True), (
            f"{name}: the payload the pipeline reported differs from the pre-step "
            "state.pop (NaN padding in the same slots counts as equal)"
        )


def test_setup_does_not_call_ask(tn, ask_pipeline):
    """``setup`` is host-only: no build, so the traced graph's ``ask`` never runs."""
    pipeline = ask_pipeline(_ask_algorithm_class(tn))
    state = pipeline.setup()

    assert pipeline.algorithm.ask_calls == 0, (
        "setup() must not build the generation graph, so the in-graph ask hook "
        f"cannot have run; ask_calls={pipeline.algorithm.ask_calls}"
    )
    assert pipeline.builds == 0, (
        f"setup() must build nothing; Pipeline.builds={pipeline.builds}"
    )
    assert _generation(tn, state) == 0, "the initial state starts at generation 0"
    assert state.pop.node_keys.shape[0] == POP_SIZE, (
        "setup() must still register the whole population, got leading axis "
        f"{state.pop.node_keys.shape[0]} for pop_size={POP_SIZE}"
    )


def test_one_build_calls_ask_exactly_once(tn, ask_pipeline):
    """The first ``step`` builds the graph, and the build makes exactly one ``ask`` call."""
    pipeline = ask_pipeline(_ask_algorithm_class(tn), generation_limit=2)
    state = pipeline.setup()

    state, _previous_pop, fitness = pipeline.step(state)

    assert pipeline.builds == 1, (
        f"the first step of a fresh shape must build once, got {pipeline.builds}"
    )
    assert pipeline.algorithm.ask_calls == 1, (
        "the traced generation kernel calls ask once, so the first build must "
        f"produce exactly one call, got {pipeline.algorithm.ask_calls}"
    )
    assert pipeline.algorithm.ask_calls == pipeline.builds, (
        "ask runs at build time, so its call count must equal Pipeline.builds: "
        f"{pipeline.algorithm.ask_calls} calls for {pipeline.builds} build(s)"
    )
    assert np.asarray(fitness).shape == (POP_SIZE,), (
        f"step must report one fitness per individual, got shape {np.asarray(fitness).shape}"
    )
    assert _generation(tn, state) == 1, "one step advances the counter to 1"


def test_a_second_step_of_the_same_shape_does_not_ask_again(tn, ask_pipeline):
    """A reused executable re-runs the *traced* call: no second build, no second ``ask``."""
    pipeline = ask_pipeline(_ask_algorithm_class(tn), generation_limit=2)
    state = pipeline.setup()

    state, _first_pop, first_fitness = pipeline.step(state)
    state, _second_pop, second_fitness = pipeline.step(state)

    assert pipeline.builds == 1, (
        "a second generation of the same shape signature must reuse the cached "
        f"executable, got {pipeline.builds} builds"
    )
    assert pipeline.algorithm.ask_calls == 1, (
        "ask is inside the compiled graph, so a reused executable must not call it "
        f"again; got {pipeline.algorithm.ask_calls} calls after two generations"
    )
    assert pipeline.algorithm.ask_calls == pipeline.builds, (
        "the invariant is ask_calls == builds, never one call per step: "
        f"{pipeline.algorithm.ask_calls} calls, {pipeline.builds} build(s), "
        "2 generations"
    )
    assert _generation(tn, state) == 2, "two steps advance the counter to 2"
    assert not np.array_equal(first_fitness, second_fitness), (
        "the second generation must have evaluated the population the first one "
        "bred, not replayed the first one's fitness"
    )


def test_the_ask_result_is_what_gets_evaluated(tn, ask_pipeline):
    """Zeroing the attributes inside ``ask`` drives every fitness to exactly ``-0.25``.

    The control is the unmodified hook at the same seed: same initial population,
    ordinary fitness values, so ``-0.25`` can only come from the zeroed payload
    reaching ``transform``/``evaluate``.
    """
    control = ask_pipeline(_ask_algorithm_class(tn), generation_limit=1)
    control_state = control.setup()
    _control_state, _control_pop, control_fitness = control.step(control_state)

    zeroing = ask_pipeline(
        _ask_algorithm_class(tn, zero_attributes=True), generation_limit=1
    )
    zeroing_state = zeroing.setup()
    zeroing_state, _zeroing_pop, zeroed_fitness = zeroing.step(zeroing_state)

    assert zeroing.algorithm.ask_calls == 1, (
        "the zeroing hook must still have run exactly once for the one build, got "
        f"{zeroing.algorithm.ask_calls}"
    )
    np.testing.assert_allclose(
        zeroed_fitness,
        ZEROED_READOUT_FITNESS,
        atol=1e-6,
        err_msg=(
            "an all-zero attribute payload reads out sigmoid(0) = 0.5 for every "
            "sample, whose mse against XOR's [0, 1, 1, 0] is exactly 0.25 "
            "(fitness -0.25); a different value means transform/evaluate did not "
            "receive the payload ask returned"
        ),
    )
    assert not np.allclose(control_fitness, ZEROED_READOUT_FITNESS, atol=1e-6), (
        "the unmodified hook on the same seed must produce ordinary fitness values, "
        f"got {control_fitness!r} -- otherwise the -0.25 above is not evidence "
        "about ask"
    )


def test_the_reported_previous_population_is_the_pre_step_payload(tn, ask_pipeline):
    """``step`` reports ``state.pop``, not the ``ask`` result: the documented mismatch.

    The zeroing hook makes the evaluated payload provably different from the
    registered one (``-0.25`` vs. an ordinary read-out), so a pipeline that reported
    the ask result would be caught here.
    """
    pipeline = ask_pipeline(
        _ask_algorithm_class(tn, zero_attributes=True), generation_limit=1
    )
    state = pipeline.setup()
    before = _payload_arrays(tn, state.pop)
    assert any(np.any(np.nan_to_num(field) != 0.0) for field in (before[1], before[3])), (
        "the initial payload's attribute arrays must carry real (non-zero) values, "
        "otherwise the mismatch this test proves cannot be observed"
    )

    new_state, previous_pop, fitness = pipeline.step(state)
    reported = _payload_arrays(tn, previous_pop)

    for name, expected, actual in zip(GENOME_FIELDS, before, reported):
        _assert_payload_equal(name, expected, actual)

    assert np.allclose(fitness, ZEROED_READOUT_FITNESS, atol=1e-6), (
        "the evaluated payload was the zeroed one, so the fitness must be "
        f"{ZEROED_READOUT_FITNESS}, got {fitness!r}"
    )
    assert np.any(np.nan_to_num(reported[1]) != 0.0), (
        "step's reported previous_pop must be the pre-step state.pop with its real "
        "attribute values, not the all-zero payload ask returned for evaluation"
    )
    assert _generation(tn, new_state) == 1, "one step advances the counter to 1"


def test_ask_calls_are_counted_per_pipeline_instance(tn, ask_pipeline):
    """Two pipelines of different widths each build once and count only their own call."""
    narrow = ask_pipeline(_ask_algorithm_class(tn), pop_size=POP_SIZE, generation_limit=2)
    wide = ask_pipeline(
        _ask_algorithm_class(tn), pop_size=WIDER_POP_SIZE, generation_limit=2
    )
    assert narrow.algorithm is not wide.algorithm, (
        "the two pipelines must hold their own algorithm instances, otherwise a "
        "shared counter would make this test vacuous"
    )
    narrow_state = narrow.setup()
    wide_state = wide.setup()

    narrow_state, _narrow_pop, narrow_fitness = narrow.step(narrow_state)
    assert narrow.algorithm.ask_calls == 1, (
        f"the {POP_SIZE}-wide pipeline must have asked once, got "
        f"{narrow.algorithm.ask_calls}"
    )
    assert wide.algorithm.ask_calls == 0, (
        "stepping one pipeline must not touch another instance's counter, got "
        f"{wide.algorithm.ask_calls} calls on the {WIDER_POP_SIZE}-wide pipeline"
    )
    assert np.asarray(narrow_fitness).shape == (POP_SIZE,), (
        f"the narrow pipeline reports {POP_SIZE} fitness values, got "
        f"{np.asarray(narrow_fitness).shape}"
    )

    wide_state, _wide_pop, wide_fitness = wide.step(wide_state)
    assert wide.algorithm.ask_calls == 1, (
        f"the {WIDER_POP_SIZE}-wide pipeline must have asked once, got "
        f"{wide.algorithm.ask_calls}"
    )
    assert wide.builds == 1 and narrow.builds == 1, (
        "each width is its own shape signature and therefore its own single "
        f"build; got narrow={narrow.builds}, wide={wide.builds}"
    )
    assert np.asarray(wide_fitness).shape == (WIDER_POP_SIZE,), (
        f"the wide pipeline reports {WIDER_POP_SIZE} fitness values, got "
        f"{np.asarray(wide_fitness).shape}"
    )

    narrow_state, _narrow_pop_2, _narrow_fitness_2 = narrow.step(narrow_state)
    wide_state, _wide_pop_2, _wide_fitness_2 = wide.step(wide_state)
    assert narrow.algorithm.ask_calls == 1 == narrow.builds, (
        "a second generation on the narrow pipeline must add neither an ask call "
        f"nor a build; got {narrow.algorithm.ask_calls} calls for "
        f"{narrow.builds} build(s)"
    )
    assert wide.algorithm.ask_calls == 1 == wide.builds, (
        "a second generation on the wide pipeline must add neither an ask call nor "
        f"a build; got {wide.algorithm.ask_calls} calls for {wide.builds} build(s)"
    )


def test_auto_run_asks_once_for_the_whole_run(tn, ask_pipeline):
    """``auto_run`` pays one build, so a multi-generation run sees exactly one call."""
    pipeline = ask_pipeline(_ask_algorithm_class(tn), generation_limit=2)
    state = pipeline.setup()

    state, best = pipeline.auto_run(state)

    assert _generation(tn, state) == 2, (
        "the target is unreachable, so auto_run must consume both generations, got "
        f"{_generation(tn, state)}"
    )
    assert best is not None, "a finite XOR fitness must be reported as a best genome"
    assert pipeline.builds == 1, (
        f"auto_run over one shape must build once, got {pipeline.builds}"
    )
    assert pipeline.algorithm.ask_calls == 1, (
        "the two generations share one compiled graph, so ask must have run only at "
        f"build time; got {pipeline.algorithm.ask_calls} calls"
    )
    assert pipeline.algorithm.ask_calls == pipeline.builds, (
        "ask_calls must equal builds (one per build, never one per generation): "
        f"{pipeline.algorithm.ask_calls} calls for {pipeline.builds} build(s)"
    )
