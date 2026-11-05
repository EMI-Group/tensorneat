"""The documented reporting tail of a run: ``auto_run``'s best fed to ``Pipeline.show`` (marker ``slow``).

What this module proves
-----------------------
Every shipped example (``examples/func_fit/*.py``, ``examples/rl/cartpole.py``) and
the ``README`` end with the same two lines::

    state, best = pipeline.auto_run(state)
    pipeline.show(state, best)

Nothing else in ``test/`` executes ``Pipeline.show`` at all, so until now a
regression in that path -- the batch-first ``best_genome`` that ``_analyse``
stores, the ``host_payload._one_individual`` promotion and rank validation it
goes through, the ``_constant_act_fn`` payload baked as graph constants, the
one-off schedule derivation -- would only have shown up in the examples.
This module locks the path end to end: the report is printed, it is the XOR
report, and both spellings the pipeline documents (the one-individual batch and
the bare per-individual rows a checkpoint stores) are accepted and describe the
same individual.

The three assertions and their oracles
--------------------------------------
* **``auto_run``'s best is directly showable.** The oracle is the problem's own
  documented report shape (``problem.func_fit.FuncFit.show``: one
  ``input: ... target: ... predict: ...`` row per dataset sample plus one
  ``loss:`` line for the individual), not a copied log. ``XOR`` has four
  samples, so four rows and one loss line are required whatever the random
  genome predicts; the loss value is only asserted to be finite, never equal to
  a recorded number.
* **The batch-first spelling is what makes it work.** The oracle is the static
  shape signature ``Pipeline`` documents for ``best_genome`` -- ``state.pop`` at
  ``P = 1``: ``node_keys (1, max_nodes)``, ``node_attrs (1, max_nodes, A)``,
  ``conn_keys (1, max_conns, key_width)``, ``conn_attrs (1, max_conns, W)``.
  Stripping the leading axis reproduces the bare rows of
  ``genomes/best_genome.npz``; both halves are shown, and the two reports must be
  textually identical because they describe the same genome.
* **``show_problem_details=True`` reports every generation without aborting.**
  ``Pipeline._analyse`` builds the problem's diagnostics arguments -- the
  constant ``act_fn`` and the host schedule -- regardless of what the hook does,
  so a run with the flag on exercises that path once per generation. The oracle
  is the pipeline's own hook contract, observed by wrapping the two instance
  methods: ``algorithm.show_details`` once with ``(state, (pop_size,) fitness)``
  and ``problem.show_details`` once with ``(state, key, act_fn, schedule)``, with
  ``detail_failures`` empty. A failing diagnostics hook is *printed* as
  ``<label> failed (...); continuing the run`` rather than raised, so the run
  must also finish, print ``Generation: 1,`` and print no such line.
* **A missing payload is a ``TensorNEATError``, not a ``TypeError``.**
  ``auto_run`` documents ``best_genome`` as a payload *or* ``None`` when no
  generation produced a finite fitness, so ``show(state, None)`` is a real caller
  mistake that must be reported in the package's own error type, naming the two
  accepted spellings and pointing at ``auto_run(state)``/``best_genome``. The
  same test then reports a legitimate payload, because a guard that rejected the
  good argument too would "fix" the crash by removing the feature.

Budgets (measured, never asserted)
----------------------------------
Every evolving test is ``pop_size=16`` × ``generation_limit=1`` with
``fitness_target=1.0`` (unreachable: ``XOR`` fitness is ``-mse <= 0``), so exactly
one generation is consumed and no convergence is claimed.
Two ``auto_run`` calls are made in total (one per module-level case), each of
which compiles once; the third case reuses a fresh pipeline of the same shapes.
The missing-payload case runs no generation at all: it calls ``setup()`` and
reports the *initial* population's first individual, so it pays only the one-off
schedule build every ``show`` pays.
No test reads a clock (``test/CONTEXT.md``: a flaky timing test is a bug in the
test).

Why this file has no skip of its own
------------------------------------
``./conftest.py``'s autouse probe (:func:`~.conftest.require_evoxc_pipeline`)
requires every documented evoxc-pipeline feature and fails with the exact list
of anything missing.
``tensorneat`` is never imported at module scope: the package arrives through the
``tn``/``tiny_pipeline`` fixtures, exactly as in every other module of this
directory.
"""

from __future__ import annotations

import numpy as np
import pytest

from .conftest import TINY_CONFIG

pytestmark = [pytest.mark.slow]

#: The static width of the node rows ``TINY_CONFIG`` builds (``max_nodes``).
MAX_NODES = TINY_CONFIG["max_nodes"]  # 8

#: The static width of the connection rows ``TINY_CONFIG`` builds (``max_conns``).
MAX_CONNS = TINY_CONFIG["max_conns"]  # 16

#: The four field names of ``genome.utils.GenomeTensors``, in payload order.
GENOME_FIELDS = ("node_keys", "node_attrs", "conn_keys", "conn_attrs")

#: The number of samples ``XOR`` reports on: the four boolean rows of its dataset.
XOR_SAMPLES = 4

#: A configuration whose run always consumes the whole generation budget.
#:
#: ``XOR`` fitness is ``-mse <= 0``, so ``fitness_target=1.0`` is unreachable and
#: ``auto_run`` advances exactly ``generation_limit`` generations.
FIXED_LENGTH = {"fitness_target": 1.0}

#: The configuration of every pipeline in this module: one generation, nothing more.
ONE_GENERATION = {"generation_limit": 1, **FIXED_LENGTH}


def _payload_shapes(payload) -> tuple[tuple[int, ...], ...]:
    """Return the host-visible shapes of a four-leaf gene payload, in payload order.

    Args:
        payload: A ``genome.utils.GenomeTensors``, a four-element tuple of
            ``evoxc.Tensor``/``numpy.ndarray`` leaves, or any other four-element
            sequence of gene arrays (the two documented spellings ``show``
            accepts).

    Returns:
        The four ``tuple`` shapes, so a test can compare against the configured
        static signature or against the same payload with its leading axis
        stripped without reaching into ``evoxc``.

    Raises:
        AssertionError: If ``payload`` does not carry exactly the four documented
            gene arrays; guessing an order would assert on the wrong arrays.
    """
    assert len(payload) == 4, (
        "expected a four-array genome payload "
        f"{GENOME_FIELDS}, got {len(payload)} array(s) of {type(payload).__name__}"
    )
    return tuple(tuple(int(size) for size in np.asarray(field).shape) for field in payload)


def _report_lines(report: str, prefix: str) -> list[str]:
    """The lines of a captured report that start with ``prefix``.

    Args:
        report: The captured stdout of ``Pipeline.show``.
        prefix: The documented line prefix (``"input:"`` or ``"loss:"``).

    Returns:
        The matching lines in print order, so a caller can count and inspect
        them without re-parsing the whole report.
    """
    return [line for line in report.splitlines() if line.startswith(prefix)]


def test_auto_run_best_is_directly_showable(tiny_pipeline, run_pipeline, capsys):
    """Invariant: ``state, best = auto_run(state)`` then ``show(state, best)`` prints the XOR report.

    Oracle: ``problem.func_fit.FuncFit.show``'s documented report -- one
    ``input: <sample>, target: <target>, predict: <value>`` row per dataset
    sample, then one ``loss: <value>`` line for the individual. ``XOR``'s
    dataset is its four boolean rows, so the counts are closed form; the values
    are not, because they come from the random genome the one generation picked.

    Assertion: ``auto_run`` returns a non-``None`` best; ``show`` returns
    ``None``; exactly four rows are printed, each carrying a ``target:`` and a
    ``predict:`` value; exactly one ``loss:`` line is printed and its number is
    finite (a dead topology would print ``nan``, which no report of a selected
    best may contain). A ``show`` that raised, printed nothing, or reported a
    different sample count fails with a message quoting the report.

    Notes:
        The capture is cleared after the run so only ``show``'s output is
        inspected: ``is_detailed=True`` makes ``auto_run`` print its own
        per-generation report, which is a different contract (pinned by
        :func:`test_show_problem_details_reports_every_generation_without_aborting`
        below, not here).
        ``generation_limit=1`` with an unreachable target keeps the run at one
        compile and one generation.

    Owner: ``pipeline.Pipeline.auto_run`` (the batched ``best_genome``),
    ``pipeline.Pipeline.show``, ``problem.func_fit.func_fit.FuncFit.show``.
    """
    pipeline = tiny_pipeline(**ONE_GENERATION)
    state, best = run_pipeline(pipeline)

    assert best is not None, (
        "auto_run must report a best genome after a generation with a valid "
        "fitness; show() has nothing to print otherwise"
    )
    capsys.readouterr()  # drop the run's own per-generation report

    returned = pipeline.show(state, best)
    report = capsys.readouterr().out

    assert returned is None, (
        "Pipeline.show must print and return None (it forwards the problem's "
        f"host-side report), got {returned!r}"
    )

    rows = _report_lines(report, "input:")
    assert len(rows) == XOR_SAMPLES, (
        f"the XOR report must print one row per dataset sample ({XOR_SAMPLES}), "
        f"got {len(rows)}; captured report was {report!r}"
    )
    assert all("target:" in line and "predict:" in line for line in rows), (
        "every reported row must carry the input, the target and the prediction "
        f"of one sample; captured rows were {rows!r}"
    )

    losses = _report_lines(report, "loss:")
    assert len(losses) == 1, (
        "the report of one individual must end with exactly one `loss:` line, "
        f"got {len(losses)}; captured report was {report!r}"
    )
    loss = float(losses[0].split(":", 1)[1])
    assert np.isfinite(loss), (
        "the reported loss of the returned best genome must be finite; a NaN "
        f"means the reported genome has no live path to its outputs, got {loss!r}"
    )


def test_the_batch_first_spelling_is_what_show_accepts(
    tiny_pipeline, run_pipeline, capsys
):
    """Invariant: the best genome is a one-individual batch, and its bare rows are accepted too.

    Oracle: the static signature ``Pipeline`` documents -- shapes are fixed at
    build time, so the reported best is ``state.pop`` at ``P = 1``:
    ``node_keys (1, max_nodes)``, ``node_attrs (1, max_nodes, A)``,
    ``conn_keys (1, max_conns, key_width)``, ``conn_attrs (1, max_conns, W)``.
    ``host_payload._one_individual`` documents that the bare per-individual rows
    a checkpoint stores (``node_keys (max_nodes,)``, ...) are promoted to that
    batch, and the two spellings are the same genome, so their reports must
    agree character for character.

    Assertion: the best's four leaves are a ``GenomeTensors``-shaped payload with
    a leading axis of 1 and the configured row widths; stripping that axis gives
    exactly the bare rows; ``show`` prints a full four-sample report for the
    batch **and** for ``tuple(field[0] for field in best)``; the second report is
    identical to the first. A regression that only accepted the batch spelling
    fails on the raise, one that rewrote the bare rows to a different individual
    fails on the text comparison, and one that dropped the population axis from
    ``best_genome`` fails on the shape assertions.

    Notes:
        Both ``show`` calls are on the same pipeline and state, so the report is
        deterministic (func fit draws no random numbers); the equality is exact,
        with no tolerance and no normalisation.

    Owner: ``pipeline.Pipeline._analyse`` (the batched ``best_genome``),
    ``pipeline.Pipeline.show``, ``host_payload._one_individual``.
    """
    pipeline = tiny_pipeline(**ONE_GENERATION)
    state, best = run_pipeline(pipeline)
    capsys.readouterr()  # drop the run's own per-generation report

    assert getattr(best, "_fields", None) == GENOME_FIELDS, (
        "the reported best genome must be a GenomeTensors-shaped payload with the "
        f"fields {GENOME_FIELDS}, got {type(best).__name__} with fields "
        f"{getattr(best, '_fields', None)!r}"
    )
    shapes = _payload_shapes(best)
    assert shapes[0] == (1, MAX_NODES), (
        "the reported best genome must be a one-individual batch of "
        f"max_nodes={MAX_NODES} node rows, i.e. node_keys {(1, MAX_NODES)}, got "
        f"{shapes[0]}"
    )
    assert shapes[1][:2] == (1, MAX_NODES), (
        "node_attrs must be the batch-first (1, max_nodes, attr_width) array, got "
        f"{shapes[1]}"
    )
    assert shapes[2][:2] == (1, MAX_CONNS) and len(shapes[2]) == 3, (
        "conn_keys must be the batch-first (1, max_conns, key_width) array with "
        f"max_conns={MAX_CONNS}, got {shapes[2]}"
    )
    assert shapes[3][:2] == (1, MAX_CONNS), (
        "conn_attrs must be the batch-first (1, max_conns, attr_width) array, got "
        f"{shapes[3]}"
    )

    bare = tuple(np.asarray(field)[0] for field in best)
    bare_shapes = _payload_shapes(bare)
    assert bare_shapes == tuple(shape[1:] for shape in shapes), (
        "the leading axis of the batch is the population axis, so stripping it "
        f"must give the bare rows a checkpoint stores; batches {shapes} versus "
        f"bare rows {bare_shapes}"
    )

    pipeline.show(state, best)
    batched_report = capsys.readouterr().out
    pipeline.show(state, bare)
    bare_report = capsys.readouterr().out

    assert batched_report.count("predict:") == XOR_SAMPLES, (
        f"the batch-first report must cover all {XOR_SAMPLES} XOR samples, got "
        f"{batched_report!r}"
    )
    assert bare_report == batched_report, (
        "show() must accept the bare per-individual rows as the same individual "
        "and print the same report; batch-first printed "
        f"{batched_report!r}, bare rows printed {bare_report!r}"
    )


def test_a_missing_payload_is_rejected_and_a_valid_one_still_prints(
    tiny_pipeline, tn, capsys
):
    """Invariant: ``show(state, None)`` raises ``TensorNEATError``; a real payload still reports.

    Oracle: ``Pipeline.auto_run`` documents ``best_genome`` as a one-individual
    batch **or** ``None`` (a run where no generation produced a finite fitness),
    so ``None`` is a documented value a caller can hand to ``show`` by accident.
    The package's own error type and an actionable message are the contract: the
    accepted spellings ``(1, max_nodes)``/``(max_nodes,)`` plus the documented
    source of a legitimate argument (``auto_run(state)``'s ``best_genome``). The
    problem's report shape -- four ``input:`` rows for ``XOR``'s four samples plus
    one ``loss:`` line -- is the oracle for the valid call.

    Assertion: ``show(state, None)`` raises ``tn.common.TensorNEATError`` (a bare
    ``TypeError`` from ``list(None)`` fails the type check), its message contains
    both spellings and ``auto_run``/``best_genome``, and nothing is printed
    before the raise. Then ``show`` on the initial population's first individual
    -- a bare per-individual payload, the other documented spelling -- prints the
    full four-sample report, so the guard rejects only the missing argument.

    Notes:
        No generation is run: ``setup()`` is the whole setup path here, so this
        case adds no evolution compile; the payload is ``state.pop`` at
        ``P = pop_size``, sliced to one individual with the package's own
        ``common.to_numpy`` (``evoxc.Tensor`` has no ``__array__``). The values in
        the report are the initial random genome's, so only the counts are
        asserted.

    Owner: ``pipeline.Pipeline.show``, ``host_payload._one_individual``.
    """
    pipeline = tiny_pipeline(**ONE_GENERATION)
    state = pipeline.setup()
    capsys.readouterr()  # setup prints nothing, but a report would be caught below

    with pytest.raises(tn.common.TensorNEATError) as excinfo:
        pipeline.show(state, None)

    message = str(excinfo.value)
    for token in ("(1, max_nodes)", "(max_nodes,)", "auto_run", "best_genome"):
        assert token in message, (
            "a missing payload must be reported with an actionable message: it "
            f"has to name {token!r} so the caller knows what to pass; message was "
            f"{message!r}"
        )
    assert capsys.readouterr().out == "", (
        "a rejected show call must not print part of a report for a genome it "
        "does not have"
    )

    bare = tuple(tn.common.to_numpy(field)[0] for field in state.pop)
    assert pipeline.show(state, bare) is None, (
        "a valid bare per-individual payload must still be reported and return "
        "None"
    )
    report = capsys.readouterr().out

    assert len(_report_lines(report, "input:")) == XOR_SAMPLES, (
        f"the report of the accepted payload must cover all {XOR_SAMPLES} XOR "
        f"samples, got {report!r}"
    )
    assert len(_report_lines(report, "loss:")) == 1, (
        f"the accepted payload must be reported with exactly one `loss:` line, "
        f"got {report!r}"
    )


def test_show_problem_details_reports_every_generation_without_aborting(
    tiny_pipeline, run_pipeline, capsys
):
    """Invariant: ``show_problem_details=True`` reports the generation and never aborts the run.

    Oracle: ``Pipeline._analyse``'s documented diagnostics contract -- with the
    flag on it calls ``problem.show_details`` with an ``act_fn`` that bakes the
    generation's payload as ``evoxc.constant`` values and a schedule derived
    host-side, once per generation. ``FuncFit`` inherits the base problem's
    documented no-op hook, so the observable report is the pipeline's own
    ``Generation: <n>, Cost time: ...`` line; what this test exercises is the
    argument construction that runs regardless of the hook, which is the
    ``_constant_act_fn`` path no other test covers. A hook that raised is
    *printed* as ``<label> failed (...); continuing the run`` instead of
    propagating, so a silent failure is visible in the captured report.

    Assertion: each of the two documented hooks is called **exactly once** for the
    one consumed generation, recorded by wrapping the instance methods
    (``pipeline.algorithm.show_details`` / ``pipeline.problem.show_details``) and
    delegating to the originals: the algorithm hook receives ``(state, fitness)``
    with a ``(pop_size,)`` fitness, and the problem hook receives four positional
    arguments ``(state, key, act_fn, schedule)`` with a callable ``act_fn`` — the
    positive witness that :meth:`Pipeline._analyse`'s ``show_problem_details``
    branch, ``_constant_act_fn`` and ``_schedule_of`` really ran.
    ``pipeline.detail_failures`` stays empty. The one-generation ``auto_run`` also
    reports ``Generation: 1,`` and reaches ``Generation limit reached!`` (so
    exactly one generation was consumed), no line matches the documented failure
    format, and the run still returns a best genome, i.e. the diagnostics did not
    disturb the evolution loop.

    Notes:
        The flag is the only difference from the other pipelines here, so a
        failure can only come from the diagnostics path. The target is
        unreachable (``fitness_target=1.0``) so the loop is not cut short by
        convergence. Wrapping the hooks is legal host-side instrumentation: the
        hooks are reporting-only Python callables, never graph operations.

    Owner: ``pipeline.Pipeline._analyse`` (the ``show_problem_details`` branch),
    ``pipeline.Pipeline._constant_act_fn``, ``pipeline.Pipeline._details``.
    """
    pipeline = tiny_pipeline(show_problem_details=True, **ONE_GENERATION)

    # Positive witness that the diagnostics branch really ran: wrap the two
    # instance hooks (host-side objects, no graph involved) and record every
    # invocation before delegating to the original.
    calls: dict[str, list[tuple[tuple[object, ...], dict[str, object]]]] = {
        "algorithm.show_details": [],
        "problem.show_details": [],
    }
    original_algorithm_show_details = pipeline.algorithm.show_details
    original_problem_show_details = pipeline.problem.show_details

    def recording_algorithm_show_details(*args: object, **kwargs: object) -> object:
        calls["algorithm.show_details"].append((args, kwargs))
        return original_algorithm_show_details(*args, **kwargs)

    def recording_problem_show_details(*args: object, **kwargs: object) -> object:
        calls["problem.show_details"].append((args, kwargs))
        return original_problem_show_details(*args, **kwargs)

    pipeline.algorithm.show_details = recording_algorithm_show_details
    pipeline.problem.show_details = recording_problem_show_details

    state, best = run_pipeline(pipeline)
    report = capsys.readouterr().out

    algorithm_calls = calls["algorithm.show_details"]
    assert len(algorithm_calls) == 1, (
        "the one consumed generation must call algorithm.show_details exactly "
        f"once, got {len(algorithm_calls)} call(s)"
    )
    algorithm_args, algorithm_kwargs = algorithm_calls[0]
    assert not algorithm_kwargs and len(algorithm_args) == 2, (
        "algorithm.show_details must be called as (state, fitness), got "
        f"{len(algorithm_args)} positional and {sorted(algorithm_kwargs)} keyword "
        "argument(s)"
    )
    algorithm_fitness = np.asarray(algorithm_args[1])
    assert algorithm_fitness.shape == (pipeline.pop_size,), (
        f"algorithm.show_details must receive the ({pipeline.pop_size},) fitness "
        f"of the evaluated generation, got shape {algorithm_fitness.shape}"
    )

    problem_calls = calls["problem.show_details"]
    assert len(problem_calls) == 1, (
        "the one consumed generation must call problem.show_details exactly once, "
        f"got {len(problem_calls)} call(s)"
    )
    problem_args, problem_kwargs = problem_calls[0]
    assert not problem_kwargs and len(problem_args) == 4, (
        "problem.show_details must be called as (state, key, act_fn, schedule), "
        f"got {len(problem_args)} positional and {sorted(problem_kwargs)} keyword "
        "argument(s)"
    )
    assert callable(problem_args[2]), (
        "the third problem.show_details argument must be the _constant_act_fn "
        f"callable, got {type(problem_args[2]).__name__}"
    )
    assert pipeline.detail_failures == {}, (
        "neither diagnostics hook may raise: detail_failures must stay empty, got "
        f"{pipeline.detail_failures}"
    )

    assert "Generation: 1," in report, (
        "show_problem_details=True must still print the one consumed generation "
        f"of the run; captured report was {report!r}"
    )
    assert "Generation limit reached!" in report, (
        "a one-generation budget on an unreachable target must be consumed "
        f"exactly, i.e. report the generation limit; captured report was {report!r}"
    )

    failures = [line for line in report.splitlines() if " failed (" in line]
    assert not failures, (
        "with show_problem_details=True both diagnostics hooks (the problem's "
        "report and the algorithm's) must run to completion; a hook that raised "
        "is printed as `<label> failed (...); continuing the run` instead of "
        f"aborting, and these lines appeared: {failures!r}"
    )
    assert best is not None, (
        "diagnostics must not disturb the evolution loop: the run must still "
        "report a best genome"
    )
