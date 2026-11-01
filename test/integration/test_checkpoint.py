"""Checkpointing: ``is_save=True`` leaves a loadable record of the run (marker ``slow``).

What this module proves
-----------------------
``Pipeline(..., is_save=True, save_dir=tmp_path)`` is the only way a user keeps a
run, so the artefacts it writes are part of the public contract, not a debugging
convenience. A short run (``generation_limit=2``) must leave:

* ``<save_dir>/config.txt`` — the configuration of the run, written once by
  ``setup`` (the legacy layout is ``json.dumps(show_config(), indent=4)``), so a
  reader can tell what produced the log;
* ``<save_dir>/log.txt`` — one line per generation under a header row naming the
  columns (``Generation,Max,Min,Mean,Std,Cost Time``);
* ``<save_dir>/genomes/best_genome.npz`` — the best genome and its fitness,
  written when ``auto_run`` returns (plus, per the legacy layout, a
  ``genomes/<generation>.npz`` snapshot per generation).

The npz round trip
------------------
``best_genome.npz`` is read back with plain ``numpy.load`` — no tensorneat code —
and must yield a genome payload the *pipeline* accepts: the same static row counts
(``max_nodes`` node rows, ``max_conns`` connection rows), the gene layout's
attribute widths, and a recorded fitness that re-scores to itself when the loaded
payload is batched to one individual and pushed through ``algorithm.transform`` →
``algorithm.forward`` → ``problem.evaluate`` (``./conftest.evaluate_population``).
A checkpoint that only *looks* right (right file names, wrong columns) fails here.

One ambiguity, resolved by not depending on it
----------------------------------------------
The pre-rewrite npz stored two dense rows per gene (``nodes``/``conns``) because a
gene row was a single mixed-type array; the rewrite splits every gene into a key
array and an attribute array (``src/tensorneat/genome/CONTEXT.md``: "there is no
bitcast", so the payload is four arrays). The documented checkpoint content is
"the best genome and its fitness", not a column layout, so
:func:`test_the_best_genome_npz_round_trips_into_a_payload` accepts either naming
(``node_keys``/``node_attrs``/``conn_keys``/``conn_attrs`` or ``nodes``/``conns``)
and asserts the *shapes, the ``fitness`` entry and the re-scored value*, which are
identical under both. The resolved layout is reported to the parent rather than
pinned here.

Budget (measured, not asserted)
-------------------------------
Two generations at ``pop_size=16``, one run per test (three runs total): budgeted
at a few seconds on the numpy reference interpreter, plus a few kilobytes of
``tmp_path`` output. The measured wall time is recorded here in Phase 2, and
``tmp_path`` is removed by pytest, so nothing leaks between tests.

Why nothing here is a module-scope import
-----------------------------------------
``import tensorneat`` still raises ``NotImplementedError`` while ``src/`` is a
design skeleton, so every input arrives as a fixture resolved through
``support.lazy.require`` (``./conftest.py``) and the stub body reports the owner.
"""

from __future__ import annotations

import pytest

pytestmark = [pytest.mark.slow]

#: Two generations, always consumed (XOR fitness is ``-mse <= 0 < 1.0``).
TWO_GENERATIONS = {"generation_limit": 2, "fitness_target": 1.0}


def test_is_save_writes_the_config_the_log_and_the_best_genome(
    tiny_pipeline, run_pipeline, tmp_path
):
    """Invariant: ``is_save=True`` creates ``config.txt``, ``log.txt`` and ``genomes/best_genome.npz``.

    Oracle: the documented directory layout, checked as a filesystem fact with no
    tensorneat code: ``<save_dir>/config.txt`` (non-empty and parseable as JSON),
    ``<save_dir>/log.txt`` (whose first line is the documented
    ``Generation,Max,Min,Mean,Std,Cost Time`` header, ``./conftest.LOG_COLUMNS``),
    and ``<save_dir>/genomes/best_genome.npz``.

    Assertion: all three paths exist and are non-empty after a two-generation
    ``auto_run`` into ``save_dir=tmp_path``; ``json.loads(config.txt)`` yields a
    mapping (``show_config`` of the registered objects), and the log's header row
    matches the documented column names exactly. The per-generation snapshots
    ``genomes/<generation>.npz`` are asserted as present for every logged
    generation, since that is the legacy layout a restore workflow reads.

    Notes:
        ``save_dir`` is passed explicitly. The documented default derives a
        timestamped directory from the current time, which this suite must never
        touch (no wall-clock dependence, ``test/CONTEXT.md``), and ``tmp_path``
        keeps the run out of the repository.

    Owner: ``pipeline.Pipeline.setup`` (``config.txt``/``log.txt``),
    ``pipeline.Pipeline.auto_run`` (``best_genome.npz``).
    """
    pytest.skip(
        "not implemented: is_save writes config.txt, log.txt and "
        "genomes/best_genome.npz (owner: pipeline.Pipeline.setup, "
        "pipeline.Pipeline.auto_run)"
    )


def test_a_short_run_logs_one_line_per_generation(tiny_pipeline, run_pipeline, tmp_path):
    """Invariant: ``log.txt`` has exactly one data line per generation, with the documented six columns.

    Oracle: the generation counter of the final state and the documented column
    list (``./conftest.LOG_COLUMNS``). ``NEAT`` increments ``generation`` at the
    *start* of ``tell`` (``src/tensorneat/algorithm/CONTEXT.md``, step 1 of the
    generation algebra), so the logged ``Generation`` column is a run of
    consecutive integers ending at ``int(state.generation)`` — the column is
    compared against that identity rather than against a hard-coded starting
    number.

    Assertion: after ``generation_limit=2`` the log holds a header plus exactly two
    data lines, every line splits into six comma-separated cells, the ``Generation``
    cells are strictly increasing by one and the last equals
    ``int(state.generation)``, and the ``Max`` cells are ``>= -inf`` floats that
    never decrease below the first — a run whose log line count drifts from its
    generation count is the failure this catches (a log appended twice per step, or
    a missing final line when the loop breaks on the target).

    Notes:
        The ``Cost Time`` column is parsed but never asserted on: it is wall-clock
        by construction, and the suite forbids clock assertions.

    Owner: ``pipeline.Pipeline.setup`` (header), ``pipeline.Pipeline.auto_run`` /
    ``pipeline.Pipeline.step`` (one append per generation).
    """
    pytest.skip(
        "not implemented: log.txt holds one line per generation with the "
        "documented columns (owner: pipeline.Pipeline.setup, pipeline.Pipeline.step)"
    )


def test_the_best_genome_npz_round_trips_into_a_payload(
    tiny_pipeline, run_pipeline, evaluate_population, tmp_path
):
    """Invariant: ``numpy.load`` round-trips the best genome into a payload that re-scores to its saved fitness.

    Oracle: the configured shape signature (``max_nodes`` node rows,
    ``max_conns`` connection rows, the gene layout's attribute widths) and the
    ``fitness`` entry inside the npz, re-scored with the documented kernel triple
    by ``./conftest.evaluate_population`` after batching the loaded payload to one
    individual (axis 0).

    Assertion: the npz holds the genome's arrays and a ``fitness`` scalar;
    ``np.load`` returns arrays whose padded row counts are ``max_nodes`` and
    ``max_conns`` and whose per-row widths match ``node_gene.length`` /
    ``conn_gene.length``; the recorded ``fitness`` is finite and equals the run's
    best within float32 tolerance; and the loaded payload re-scores to that same
    value, so the file round trip preserves the genome and not merely its name.

    Notes:
        Either the four-array payload naming or the legacy ``nodes``/``conns`` pair
        is accepted (module docstring: which one the rewrite stores is genome-owned
        and not documented), and the assertions are the ones both layouts share.
        Nothing here writes to ``../src``: the checkpoint layout is read, never
        imposed.

    Owner: ``pipeline.Pipeline.auto_run`` (what is written),
    ``genome.utils.GenomeTensors`` (what the loaded payload must be), and
    ``problem.func_fit.func_fit.FuncFit.evaluate`` (the re-scoring oracle).
    """
    pytest.skip(
        "not implemented: best_genome.npz round-trips into a one-individual payload "
        "that re-scores to its saved fitness (owner: pipeline.Pipeline.auto_run, "
        "genome.utils.GenomeTensors)"
    )
