"""A failing diagnostics hook is reported, recorded and survived — never swallowed and never fatal (marker ``slow``).

What this module proves
-----------------------
``Pipeline._analyse`` calls two *reporting-only* hooks each generation:
``algorithm.show_details(state, fitnesses)`` when ``is_detailed=True`` and
``problem.show_details(state, randkey, act_fn, schedule)`` when
``show_problem_details=True``. ``Pipeline._details``
(``src/tensorneat/pipeline.py``:860-876) runs each inside a guard, so a hook that
raises neither aborts the evolution nor disappears:

* it is **printed** as ``<label> failed (<Type>: <msg>); continuing the run``, so the
  failure stays visible in the run log;
* its ``(type_name, message)`` is **recorded** in ``pipeline.detail_failures``, keyed
  by the hook label — exactly ``"algorithm.show_details"`` and
  ``"problem.show_details"``;
* only the **first** failure per label is kept (``setdefault``), so a hook that breaks
  every generation cannot grow the record without bound — while every attempt is
  still printed.

The tests drive that with a ``NEAT`` subclass and an ``XOR`` subclass whose
``show_details`` count their calls and raise a message containing the call number, so
"the recorded message is the first one" and "both attempts reached stdout" are
observable facts rather than inferences. A working-hook run pins the other half of
the contract: ``detail_failures`` stays ``{}`` when nothing fails, so the record holds
genuine failures only. A third case pins the documented gating — with
``is_detailed=False`` the algorithm hook is never called at all, so its failure is (by
construction) not recorded, while the problem hook still runs and a working one still
reports.

The oracles
-----------
``state.generation`` and ``auto_run``'s best genome (the run really finished), the
hook call counters (each hook really ran once per generation), the exact recorded
``(type_name, message)`` pairs, and the captured stdout (the documented continuation
lines). Nothing is compared against a recorded number; the mutation messages are
built by the test.

Budgets (measured, never asserted)
----------------------------------
``pop_size=8``, ``max_nodes=8``, ``generation_limit=2`` and
``fitness_target=1.0`` (unreachable for ``XOR``: fitness is ``-mse <= 0``), so every
run consumes exactly two generations and builds one executable (plus the one-off
schedule derivation each ``show_problem_details`` generation pays). Nothing here
reads a clock.

Why this file has no ``tensorneat`` import at module scope
---------------------------------------------------------
The package arrives through the ``tn`` fixture; the subclasses are created inside the
tests, so a failed import is a named skip from ``support.lazy.require`` rather than a
collection error. The directory-wide autouse probe
(:func:`~.conftest.require_evoxc_pipeline`) is the only guard.
"""

from __future__ import annotations

import pytest

from .conftest import TINY_CONFIG

pytestmark = [pytest.mark.slow]

#: The genome knobs every pipeline in this module shares, from the suite's one
#: configuration: ``XOR`` arity ``(2 -> 1)`` plus the padded gene geometry.
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

#: The population of every pipeline here.
POP_SIZE = 8

#: ``species_size * min_species_size <= pop_size`` must hold: ``2 * 1 <= 8``.
SPECIES_SIZE = 2

#: One seed for every pipeline.
SEED = 0

#: The number of generations each run consumes.
GENERATIONS = 2

#: Unreachable for ``XOR`` (fitness ``-mse <= 0``): the whole budget is consumed.
UNREACHABLE_TARGET = 1.0

#: The record keys ``Pipeline._details`` uses for the two hooks.
ALGORITHM_LABEL = "algorithm.show_details"
PROBLEM_LABEL = "problem.show_details"

#: The marker every continuation line carries.
CONTINUATION = "; continuing the run"


def _failing_algorithm_class(tn):
    """Build a ``NEAT`` subclass whose ``show_details`` raises with its call number.

    Args:
        tn: The imported ``tensorneat`` package (the ``tn`` fixture); its
            ``algorithm.NEAT`` is the base class, so the population handling is the
            shipped one and only the *reporting* hook is broken.

    Returns:
        The subclass. ``show_details_calls`` counts attempts, and each call raises
        ``RuntimeError("neat report boom #<n>")``.
    """

    class FailingNEAT(tn.algorithm.NEAT):
        """``NEAT`` whose diagnostics printer always raises ``RuntimeError``."""

        def __init__(self, *args, **kwargs) -> None:
            super().__init__(*args, **kwargs)
            #: How many times the pipeline tried to print this generation's report.
            self.show_details_calls = 0

        def show_details(self, state, fitness) -> None:
            """Count the attempt, then raise a message naming it.

            Raises:
                RuntimeError: Always; the pipeline must print and record it instead
                    of propagating.
            """
            self.show_details_calls += 1
            raise RuntimeError(f"neat report boom #{self.show_details_calls}")

    return FailingNEAT


def _failing_problem_class(tn):
    """Build an ``XOR`` subclass whose ``show_details`` raises with its call number.

    Args:
        tn: The imported ``tensorneat`` package; its ``problem.XOR`` is the base
            class, so the problem itself evaluates exactly like the shipped one.

    Returns:
        The subclass. ``show_details_calls`` counts attempts, and each call raises
        ``ValueError("xor report boom #<n>")`` — a different type from the
        algorithm hook so the recorded pairs cannot be confused.
    """

    class FailingXOR(tn.problem.XOR):
        """``XOR`` whose diagnostics hook always raises ``ValueError``."""

        def __init__(self, *args, **kwargs) -> None:
            super().__init__(*args, **kwargs)
            #: How many times the pipeline tried to report on this problem.
            self.show_details_calls = 0

        def show_details(self, state, key, act_fn, transformed) -> None:
            """Count the attempt, then raise a message naming it.

            Raises:
                ValueError: Always; the pipeline must print and record it instead of
                    propagating.
            """
            self.show_details_calls += 1
            raise ValueError(f"xor report boom #{self.show_details_calls}")

    return FailingXOR


def _counting_problem_class(tn):
    """Build an ``XOR`` subclass whose ``show_details`` works and counts its calls.

    Args:
        tn: The imported ``tensorneat`` package; its ``problem.XOR`` is the base
            class.

    Returns:
        The subclass: ``show_details`` increments ``show_details_calls`` and delegates
        to the base implementation, so the hook can be *proven* to have run even
        though the shipped one prints nothing.
    """

    class CountingXOR(tn.problem.XOR):
        """``XOR`` with a working, call-counting diagnostics hook."""

        def __init__(self, *args, **kwargs) -> None:
            super().__init__(*args, **kwargs)
            #: How many times the pipeline called the (working) hook.
            self.show_details_calls = 0

        def show_details(self, state, key, act_fn, transformed) -> None:
            """Count the call and run the base hook."""
            self.show_details_calls += 1
            super().show_details(state, key, act_fn, transformed)

    return CountingXOR


@pytest.fixture(scope="module")
def detail_pipeline(tn):
    """Factory for a small ``Pipeline`` whose two hooks are supplied by the test.

    Returns:
        ``make(*, algorithm_class=None, problem_class=None, is_detailed=True,
        show_problem_details=True, generation_limit=GENERATIONS) -> Pipeline``:
        ``DefaultGenome`` (:data:`GENOME_KNOBS`, sigmoid read-out) → the requested
        algorithm class (``tn.algorithm.NEAT`` by default) → the requested problem
        class (``tn.problem.XOR(error_method="mse")`` by default) → a numpy-backend
        ``Pipeline`` at :data:`SEED` with :data:`UNREACHABLE_TARGET`. Nothing is run.

    Notes:
        Both flags default to ``True`` because the two hooks are gated on different
        flags (``is_detailed`` and ``show_problem_details``); a test that wants to
        pin one gate passes the other flag explicitly.
    """

    def make(
        *,
        algorithm_class=None,
        problem_class=None,
        is_detailed=True,
        show_problem_details=True,
        generation_limit=GENERATIONS,
    ):
        genome = tn.genome.DefaultGenome(
            **GENOME_KNOBS, output_transform=tn.common.ACT.sigmoid
        )
        algorithm = (algorithm_class or tn.algorithm.NEAT)(
            genome, pop_size=POP_SIZE, species_size=SPECIES_SIZE
        )
        problem = (problem_class or tn.problem.XOR)(error_method="mse")
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
            is_detailed=is_detailed,
            show_problem_details=show_problem_details,
        )

    return make


def _generation(tn, state) -> int:
    """Read ``state.generation`` (a 0-d float32 evoxc tensor) as a Python ``int``.

    Args:
        tn: The imported ``tensorneat`` package, for ``common.tools.to_numpy``.
        state: Any ``State`` the pipeline returned.

    Returns:
        The generation counter.
    """
    return int(tn.common.tools.to_numpy(state.generation))


def _continuation_lines(stdout: str) -> list[str]:
    """Split captured stdout into the diagnostics-failure lines it contains.

    Args:
        stdout: The text ``capsys.readouterr().out`` returned.

    Returns:
        Every line carrying the documented ``; continuing the run`` marker, in
        output order, stripped. Empty when no hook failed.
    """
    return [line.strip() for line in stdout.splitlines() if CONTINUATION in line]


def test_a_failing_diagnostics_hook_does_not_abort_the_run(tn, detail_pipeline, capsys):
    """Both hooks raise, both failures are printed and recorded, and the run finishes."""
    pipeline = detail_pipeline(
        algorithm_class=_failing_algorithm_class(tn),
        problem_class=_failing_problem_class(tn),
    )
    state = pipeline.setup()
    state, best = pipeline.auto_run(state)
    stdout = capsys.readouterr().out

    assert _generation(tn, state) == GENERATIONS, (
        "a failing reporting hook must not abort the evolution, so the unreachable "
        f"target must consume all {GENERATIONS} generations, got "
        f"{_generation(tn, state)}"
    )
    assert best is not None, (
        "XOR produced finite fitness, so auto_run must still report a best genome "
        "even though both diagnostics hooks failed"
    )
    assert set(pipeline.detail_failures) == {ALGORITHM_LABEL, PROBLEM_LABEL}, (
        "exactly the two documented hook labels must be recorded in "
        f"detail_failures, got {sorted(pipeline.detail_failures)}"
    )
    assert pipeline.detail_failures[ALGORITHM_LABEL] == (
        "RuntimeError",
        "neat report boom #1",
    ), (
        "the algorithm hook's record must be its first failure as "
        f"(type, message), got {pipeline.detail_failures[ALGORITHM_LABEL]!r}"
    )
    assert pipeline.detail_failures[PROBLEM_LABEL] == (
        "ValueError",
        "xor report boom #1",
    ), (
        "the problem hook's record must be its first failure as (type, message), "
        f"got {pipeline.detail_failures[PROBLEM_LABEL]!r}"
    )
    assert pipeline.algorithm.show_details_calls == GENERATIONS, (
        "the algorithm hook must have been attempted once per generation, got "
        f"{pipeline.algorithm.show_details_calls}"
    )
    assert pipeline.problem.show_details_calls == GENERATIONS, (
        "the problem hook must have been attempted once per generation, got "
        f"{pipeline.problem.show_details_calls}"
    )

    lines = _continuation_lines(stdout)
    assert len(lines) == 2 * GENERATIONS, (
        "every failed attempt must be printed (two hooks x "
        f"{GENERATIONS} generations), got {lines}"
    )
    assert (
        f"{ALGORITHM_LABEL} failed (RuntimeError: neat report boom #1){CONTINUATION}"
        in lines
    ), f"the documented continuation line for the algorithm hook is missing: {lines}"
    assert (
        f"{ALGORITHM_LABEL} failed (RuntimeError: neat report boom #2){CONTINUATION}"
        in lines
    ), "the second generation's algorithm failure must be printed too, not dropped"
    assert (
        f"{PROBLEM_LABEL} failed (ValueError: xor report boom #1){CONTINUATION}" in lines
    ), f"the documented continuation line for the problem hook is missing: {lines}"
    assert (
        f"{PROBLEM_LABEL} failed (ValueError: xor report boom #2){CONTINUATION}" in lines
    ), "the second generation's problem failure must be printed too, not dropped"
    assert f"Generation: {GENERATIONS}," in stdout, (
        "the run's own per-generation report must still be printed alongside the "
        f"hook failures, got: {stdout!r}"
    )


def test_the_record_keeps_the_first_failure_per_hook(tn, detail_pipeline, capsys):
    """``detail_failures`` holds the *first* message per label while stdout shows every attempt."""
    pipeline = detail_pipeline(
        algorithm_class=_failing_algorithm_class(tn),
        problem_class=_failing_problem_class(tn),
    )
    state = pipeline.setup()
    state, _best = pipeline.auto_run(state)
    stdout = capsys.readouterr().out

    assert pipeline.algorithm.show_details_calls == GENERATIONS, (
        "the hook must have failed more than once for 'first only' to mean anything, "
        f"got {pipeline.algorithm.show_details_calls} attempt(s)"
    )
    assert pipeline.problem.show_details_calls == GENERATIONS, (
        "the problem hook must have failed more than once too, got "
        f"{pipeline.problem.show_details_calls} attempt(s)"
    )

    algorithm_type, algorithm_message = pipeline.detail_failures[ALGORITHM_LABEL]
    problem_type, problem_message = pipeline.detail_failures[PROBLEM_LABEL]
    assert algorithm_message.endswith("#1") and "#2" not in algorithm_message, (
        "the record must keep the first algorithm failure, so its message ends in "
        f"'#1', got {algorithm_message!r} ({algorithm_type})"
    )
    assert problem_message.endswith("#1") and "#2" not in problem_message, (
        "the record must keep the first problem failure, so its message ends in "
        f"'#1', got {problem_message!r} ({problem_type})"
    )

    lines = _continuation_lines(stdout)
    assert any(line.endswith("neat report boom #2)" + CONTINUATION) for line in lines), (
        "the later attempt is not recorded but must still be visible on stdout, got "
        f"{lines}"
    )
    assert any(line.endswith("xor report boom #2)" + CONTINUATION) for line in lines), (
        "the later problem attempt is not recorded but must still be visible on "
        f"stdout, got {lines}"
    )


def test_working_hooks_leave_the_record_empty(tn, detail_pipeline, capsys):
    """With hooks that work, nothing is recorded and both hooks still run each generation."""
    pipeline = detail_pipeline(problem_class=_counting_problem_class(tn))
    state = pipeline.setup()
    state, best = pipeline.auto_run(state)
    stdout = capsys.readouterr().out

    assert _generation(tn, state) == GENERATIONS, (
        f"the control run must consume both generations, got {_generation(tn, state)}"
    )
    assert best is not None, "the control run must report a best genome"
    assert pipeline.detail_failures == {}, (
        "detail_failures must hold genuine failures only; a working hook leaves it "
        f"empty, got {pipeline.detail_failures!r}"
    )
    assert _continuation_lines(stdout) == [], (
        f"a successful hook prints no continuation line, got {_continuation_lines(stdout)}"
    )
    assert pipeline.problem.show_details_calls == GENERATIONS, (
        "the problem hook must have run once per generation, got "
        f"{pipeline.problem.show_details_calls}"
    )
    assert stdout.count("node counts:") == GENERATIONS, (
        "the shipped NEAT report prints one 'node counts:' line per generation, so "
        f"the algorithm hook must have run {GENERATIONS} times, got "
        f"{stdout.count('node counts:')}"
    )
    assert stdout.count("Generation: ") == GENERATIONS, (
        f"one per-generation report per generation, got {stdout.count('Generation: ')}"
    )


def test_the_algorithm_hook_is_gated_on_is_detailed(tn, detail_pipeline):
    """``is_detailed=False`` never calls the algorithm hook, so its failure is never recorded."""
    pipeline = detail_pipeline(
        algorithm_class=_failing_algorithm_class(tn),
        problem_class=_counting_problem_class(tn),
        is_detailed=False,
        show_problem_details=True,
    )
    state = pipeline.setup()
    state, best = pipeline.auto_run(state)

    assert _generation(tn, state) == GENERATIONS, (
        f"the run must still consume both generations, got {_generation(tn, state)}"
    )
    assert best is not None, "the run must still report a best genome"
    assert pipeline.algorithm.show_details_calls == 0, (
        "is_detailed=False must suppress the algorithm hook entirely, so a broken "
        f"one is never attempted, got {pipeline.algorithm.show_details_calls} call(s)"
    )
    assert pipeline.problem.show_details_calls == GENERATIONS, (
        "show_problem_details=True is a separate flag, so the problem hook must still "
        f"run once per generation, got {pipeline.problem.show_details_calls}"
    )
    assert pipeline.detail_failures == {}, (
        "a hook that is never called cannot fail, so the record must stay empty, got "
        f"{pipeline.detail_failures!r}"
    )
