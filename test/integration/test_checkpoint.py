"""Checkpointing: ``is_save=True`` leaves a loadable record of the run (marker ``slow``).

What this module proves
-----------------------
``Pipeline(..., is_save=True, save_dir=tmp_path)`` is the only way a user keeps a run, so the artefacts it writes are part of the public contract, not a debugging convenience.
A short run (``generation_limit=2``) must leave:

* ``<save_dir>/config.txt`` — the configuration of the run, written once by ``setup`` as ``json.dumps(show_config(), indent=4)``, so a reader can tell what produced the log;
* ``<save_dir>/log.txt`` — one line per generation under the header row ``Generation,Max,Min,Mean,Std,Cost Time`` (``LOG_COLUMNS``);
* ``<save_dir>/genomes/best_genome.npz`` — the best genome and its fitness, written when ``auto_run`` returns, plus a ``genomes/<generation>.npz`` snapshot per generation.

The npz round trip
------------------
``best_genome.npz`` is read back with plain ``numpy.load`` — no tensorneat code — and must yield a payload the *pipeline* accepts: the static row counts (``max_nodes`` node rows, ``max_conns`` connection rows), the split layout's attribute widths, and a recorded ``fitness`` that re-scores to itself once the loaded genome is batched to one individual and pushed through ``algorithm.transform`` → ``algorithm.forward`` → ``problem.evaluate`` (:func:`integration.conftest.evaluate_population`).
A checkpoint that only *looks* right (right file names, wrong columns) fails here.

The payload naming
------------------
A gene is stored as a key array plus an attribute array (``genome.utils.GenomeTensors``: ``node_keys``/``node_attrs``/``conn_keys``/``conn_attrs``) because there is no bitcast, so the checkpoint must hold those four arrays.
The pre-rewrite pipeline stored a mixed-dtype ``nodes``/``conns`` pair instead; a file in that naming cannot be rebuilt into a payload without inventing the bitcast, so it fails with a message naming the four arrays rather than being faked into a pass.

Budget (measured, not asserted)
-------------------------------
Two generations at ``pop_size=16``, one run per test (three runs total): a few seconds on the numpy reference interpreter, plus a few kilobytes under ``tmp_path``, which pytest removes afterwards.
No test reads a clock.

Why nothing here is a module-scope ``tensorneat`` import
--------------------------------------------------------
``tn`` and every ``support`` input arrive as fixtures resolved through ``support.lazy.require`` (``./conftest.py``), so an import that fails names a skip instead of turning this module into a collection error.
The autouse ``_evoxc_pipeline_required`` probe of ``./conftest.py`` requires the evoxc-native ``Pipeline``; it finds every documented feature and does not skip.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from integration.conftest import LOG_COLUMNS, TINY_CONFIG, fitness_history

pytestmark = [pytest.mark.slow]

#: Two generations, always consumed (XOR fitness is ``-mse <= 0 < 1.0``).
TWO_GENERATIONS = {"generation_limit": 2, "fitness_target": 1.0}

#: The four arrays a checkpoint must hold, in ``GenomeTensors`` order.
SPLIT_PAYLOAD = ("node_keys", "node_attrs", "conn_keys", "conn_attrs")

#: The mixed-dtype pair the pre-rewrite pipeline stored instead, named only in the failure message.
LEGACY_PAYLOAD = ("nodes", "conns")

#: Per-row widths of the two attribute arrays (``node_gene.attr_width`` / ``conn_gene.attr_width`` of ``genome.DefaultGenome``).
NODE_ATTR_WIDTH = 4
CONN_ATTR_WIDTH = 1


# --------------------------------------------------------------------------- #
# plain-filesystem readers (no tensorneat code in the file assertions)
# --------------------------------------------------------------------------- #


def _log_lines(save_dir: Path) -> tuple[str, list[list[str]]]:
    """Split ``<save_dir>/log.txt`` into its header line and its rows of comma-separated cells.

    Args:
        save_dir: The ``save_dir`` the pipeline wrote into (the ``tmp_path`` fixture).

    Returns:
        ``(header, rows)``: the first line verbatim and every following line split on
        ``","``, in file order.

    Raises:
        AssertionError: If the file is missing or holds no line at all. An empty log is
            indistinguishable from a run that never logged, so this fails loudly instead
            of returning an empty list.
    """
    path = Path(save_dir) / "log.txt"
    assert path.is_file(), f"Pipeline.setup must write {path}"
    lines = path.read_text().splitlines()
    assert lines, (
        f"{path} is empty: Pipeline.setup must write the header row "
        f"{','.join(LOG_COLUMNS)!r} before the first generation"
    )
    return lines[0], [line.split(",") for line in lines[1:]]


def _expected_shapes() -> dict[str, tuple[int, ...]]:
    """The documented one-genome shape of each stored gene array, keyed by array name.

    Returns:
        ``{name: shape}`` for the four :data:`SPLIT_PAYLOAD` arrays: the configured
        ``max_nodes`` node rows (``node_keys``) and ``max_conns`` connection rows
        (``conn_keys`` with their ``(src, dst)`` pair), with the node/conn gene
        attribute widths for ``node_attrs``/``conn_attrs``.
    """
    max_nodes = TINY_CONFIG["max_nodes"]
    max_conns = TINY_CONFIG["max_conns"]
    return {
        "node_keys": (max_nodes,),
        "node_attrs": (max_nodes, NODE_ATTR_WIDTH),
        "conn_keys": (max_conns, 2),
        "conn_attrs": (max_conns, CONN_ATTR_WIDTH),
    }


def _single_individual(
    array: np.ndarray, name: str, expected_shape: tuple[int, ...]
) -> np.ndarray:
    """Return one stored gene array as the single genome it holds, in ``expected_shape``.

    Args:
        array: One array loaded from ``best_genome.npz``.
        name: The array's key, used in the failure message.
        expected_shape: The documented one-genome shape of that array
            (:func:`_expected_shapes`).

    Returns:
        ``array`` itself when it already has ``expected_shape``, or ``array[0]`` when a
        batch-first writer kept the population axis with a single entry
        (``(1,) + expected_shape``) — the same genome under either spelling.

    Raises:
        AssertionError: For any other shape: a stored *population* or a narrower row
            width is a contract change, and it must not be silently reshaped into
            something scorable.
    """
    if array.shape == expected_shape:
        return array
    if array.shape == (1,) + expected_shape:
        return array[0]
    raise AssertionError(
        f"{name} in best_genome.npz must be {expected_shape} — one genome at the "
        f"configured row counts and attribute widths, optionally under a leading "
        f"population axis of 1 ({(1,) + expected_shape}) — got {array.shape}"
    )


def _one_individual_payload(state: Any, arrays: tuple[np.ndarray, ...]) -> Any:
    """Rebuild a loaded genome as a one-individual payload of ``state.pop``'s type.

    Args:
        state: The ``State`` the run ended in; ``type(state.pop)`` is the container
            the algorithm expects (``GenomeTensors`` when the payload is a namedtuple).
        arrays: The four loaded arrays, already stripped of a leading batch axis.

    Returns:
        The same container type as ``state.pop`` holding ``(1, ...)`` leaves, or the
        plain four-array tuple when that type is not a namedtuple (both spellings are
        accepted by :func:`integration.conftest.evaluate_population`).
    """
    batched = tuple(array[None, ...] for array in arrays)
    payload_type = type(state.pop)
    if getattr(payload_type, "_fields", None) is None:
        return batched
    return payload_type(*batched)


# --------------------------------------------------------------------------- #
# the artefacts of a saved run
# --------------------------------------------------------------------------- #


def test_is_save_writes_the_config_the_log_and_the_best_genome(
    tiny_pipeline, run_pipeline, tmp_path
):
    """Invariant: ``is_save=True`` creates ``config.txt``, ``log.txt`` and ``genomes/best_genome.npz``.

    Oracle: the documented directory layout, checked as a filesystem fact with no
    tensorneat code — ``<save_dir>/config.txt`` (non-empty, ``json.loads``-parseable
    mapping; the legacy layout is ``json.dumps(show_config(), indent=4)``, so any
    other format is a contract change to escalate), ``<save_dir>/log.txt`` whose first
    line is exactly the comma-joined ``LOG_COLUMNS``, and
    ``<save_dir>/genomes/best_genome.npz``.

    Assertion: all paths exist and are non-empty after a two-generation ``auto_run``
    into ``save_dir=tmp_path``; ``json.loads`` yields a non-empty mapping; the log
    header row matches the documented column names exactly; and the legacy
    per-generation snapshot ``genomes/<generation>.npz`` exists for every generation
    the log names, since that is the layout a restore workflow reads.

    Notes:
        ``save_dir`` is passed explicitly. The documented default derives a
        timestamped directory from the current time, which this suite must never touch
        (no wall-clock dependence, ``test/CONTEXT.md``), and ``tmp_path`` keeps the run
        out of the repository.

    Owner: ``pipeline.Pipeline.setup`` (``config.txt``/``log.txt``),
    ``pipeline.Pipeline.auto_run`` (``best_genome.npz``).
    """
    pipeline = tiny_pipeline(is_save=True, save_dir=tmp_path, **TWO_GENERATIONS)
    run_pipeline(pipeline)

    config_path = tmp_path / "config.txt"
    assert config_path.is_file(), f"Pipeline.setup must write {config_path}"
    config_text = config_path.read_text()
    assert config_text.strip(), (
        f"{config_path} is empty: Pipeline.setup must write "
        "json.dumps(pipeline.show_config(), indent=4)"
    )
    config = json.loads(config_text)
    assert isinstance(config, dict) and config, (
        f"{config_path} must hold the run's configuration as a JSON mapping "
        f"(the legacy layout is json.dumps(show_config(), indent=4)); "
        f"got {type(config).__name__} {config!r}"
    )

    header, rows = _log_lines(tmp_path)
    assert header == ",".join(LOG_COLUMNS), (
        f"{config_path.with_name('log.txt')} header must be the documented columns "
        f"{','.join(LOG_COLUMNS)!r}, got {header!r}"
    )

    best_path = tmp_path / "genomes" / "best_genome.npz"
    assert best_path.is_file(), (
        f"Pipeline.auto_run must write the best genome to {best_path} when is_save"
    )
    assert best_path.stat().st_size > 0, f"{best_path} is empty"

    assert rows, (
        f"{tmp_path / 'log.txt'} holds a header but no generation line, so no "
        "per-generation snapshot can be checked"
    )
    for row in rows:
        generation = int(row[0])
        snapshot = tmp_path / "genomes" / f"{generation}.npz"
        assert snapshot.is_file(), (
            f"Pipeline.auto_run must snapshot generation {generation} to {snapshot} "
            f"(the legacy per-generation layout a restore workflow reads); logged "
            f"generations are {[int(cell[0]) for cell in rows]}"
        )
        assert snapshot.stat().st_size > 0, f"{snapshot} is empty"


def test_a_short_run_logs_one_line_per_generation(
    tiny_pipeline, run_pipeline, runner, tmp_path
):
    """Invariant: ``log.txt`` has exactly one data line per generation, with the documented six columns.

    Oracle: the generation counter of the final state and the documented column list
    (``LOG_COLUMNS``). ``NEAT`` increments ``generation`` at the *start* of ``tell``
    (``src/tensorneat/algorithm/CONTEXT.md``, step 1 of the generation algebra), so
    the logged ``Generation`` column is a run of consecutive integers ending at
    ``int(state.generation)`` — compared against that identity rather than against a
    hard-coded starting number.

    Assertion: after ``generation_limit=2`` the log holds a header plus exactly two
    data lines, every line splits into six comma-separated cells, the ``Generation``
    cells increase by one and the last equals ``int(to_numpy(state.generation))``,
    and the ``Max`` cells parse as floats and never decrease — elitism keeps the
    population maximum non-decreasing. :func:`integration.conftest.fitness_history`
    must report the same generation sequence and the same maxima, so the log and the
    stepped run cannot disagree. A run whose log line count drifts from its
    generation count is the failure this catches (a line appended twice per step, or
    a missing final line when the loop breaks on the target).

    Notes:
        The ``Cost Time`` column is parsed but never asserted on: it is wall-clock by
        construction, and the suite forbids clock assertions.

    Owner: ``pipeline.Pipeline.setup`` (header), ``pipeline.Pipeline.auto_run`` /
    ``pipeline.Pipeline.step`` (one append per generation).
    """
    pipeline = tiny_pipeline(is_save=True, save_dir=tmp_path, **TWO_GENERATIONS)
    state, _best = run_pipeline(pipeline)

    header, rows = _log_lines(tmp_path)
    assert header == ",".join(LOG_COLUMNS), (
        f"log.txt header must be the documented columns {','.join(LOG_COLUMNS)!r}, "
        f"got {header!r}"
    )
    assert len(rows) == TWO_GENERATIONS["generation_limit"], (
        f"generation_limit={TWO_GENERATIONS['generation_limit']} must log one line per "
        f"generation (XOR can never reach fitness_target "
        f"{TWO_GENERATIONS['fitness_target']}); got {len(rows)} data line(s): {rows}"
    )
    for row in rows:
        assert len(row) == len(LOG_COLUMNS), (
            f"every log line must have the {len(LOG_COLUMNS)} cells {LOG_COLUMNS}, "
            f"got {len(row)} in {row}"
        )

    generations = [int(row[0]) for row in rows]
    assert np.diff(generations).tolist() == [1] * (len(generations) - 1), (
        f"the Generation column must be consecutive integers, got {generations}"
    )
    final_generation = int(runner.to_numpy(state.generation))
    assert generations[-1] == final_generation, (
        f"the last logged generation ({generations[-1]}) must be the run's final "
        f"generation ({final_generation})"
    )

    maxima = [float(row[1]) for row in rows]
    assert all(maxima[index] >= maxima[index - 1] for index in range(1, len(maxima))), (
        "elitism keeps the population maximum non-decreasing, so the Max column may "
        f"never fall, got {maxima} from {rows}"
    )

    # Parsed, never asserted on: ``Cost Time`` is wall-clock by construction, and a
    # non-numeric cell still fails here with ``ValueError``.
    _cost_times = [float(row[5]) for row in rows]

    recorded = fitness_history(pipeline)
    assert [generation for generation, _ in recorded] == generations, (
        "fitness_history must report the same generation sequence the log records, "
        f"got {[generation for generation, _ in recorded]} vs {generations}"
    )
    np.testing.assert_allclose(
        [best for _, best in recorded],
        maxima,
        rtol=1e-6,
        atol=1e-6,
        err_msg="the logged Max column and the run's fitness history disagree",
    )


def test_the_best_genome_npz_round_trips_into_a_payload(
    tiny_pipeline, run_pipeline, evaluate_population, tmp_path
):
    """Invariant: ``numpy.load`` round-trips the best genome into a payload that re-scores to its saved fitness.

    Oracle: the configured shape signature (``TINY_CONFIG``'s ``max_nodes`` node rows,
    ``max_conns`` connection rows, attribute widths ``NODE_ATTR_WIDTH``/
    ``CONN_ATTR_WIDTH``), the ``fitness`` entry inside the npz, and the re-scored
    fitness from :func:`integration.conftest.evaluate_population` after batching the
    loaded genome to one individual on axis 0.

    Assertion: the npz holds the four split arrays (``node_keys``/``node_attrs``/
    ``conn_keys``/``conn_attrs``) and a scalar ``fitness``; the node arrays have
    ``max_nodes`` rows, the connection arrays ``max_conns``, with per-row widths
    ``(NODE_ATTR_WIDTH,)`` and ``(CONN_ATTR_WIDTH,)``; the recorded ``fitness`` is
    finite, equals the best value of the run's fitness history, and is reproduced by
    re-scoring the loaded payload within ``rtol=1e-5, atol=1e-6`` — so the file
    round trip preserves the genome and not merely its name.

    Notes:
        A legacy ``nodes``/``conns`` file fails here with a message naming the four
        arrays it needs: rebuilding a mixed-dtype gene row from numpy would require a
        bitcast the rewrite does not have, and a fabricated payload would prove
        nothing. A leading population axis of size 1 is the same genome and is
        dropped (:func:`_single_individual` against :func:`_expected_shapes`).
        Nothing here writes to ``../src``: the checkpoint layout is read, never imposed.

    Owner: ``pipeline.Pipeline.auto_run`` (what is written),
    ``genome.utils.GenomeTensors`` (what the loaded payload must be), and
    ``problem.func_fit.func_fit.FuncFit.evaluate`` (the re-scoring oracle).
    """
    pipeline = tiny_pipeline(is_save=True, save_dir=tmp_path, **TWO_GENERATIONS)
    state, _best = run_pipeline(pipeline)

    npz_path = tmp_path / "genomes" / "best_genome.npz"
    assert npz_path.is_file(), f"Pipeline.auto_run must write {npz_path}"
    with np.load(npz_path) as archive:
        stored = {name: archive[name] for name in archive.files}

    missing = [name for name in SPLIT_PAYLOAD if name not in stored]
    assert not missing, (
        f"{npz_path} holds {sorted(stored)} but lacks {missing}: the checkpoint must "
        f"store the split gene payload {SPLIT_PAYLOAD} (the legacy "
        f"{LEGACY_PAYLOAD} pair is a mixed-dtype row that cannot be rebuilt without a "
        "bitcast), plus a scalar 'fitness'"
    )
    assert "fitness" in stored, (
        f"{npz_path} holds {sorted(stored)} but no 'fitness' entry; a checkpoint "
        "without the fitness it was selected for cannot be verified on load"
    )

    shapes = _expected_shapes()
    arrays = tuple(
        _single_individual(stored[name], name, shapes[name]) for name in SPLIT_PAYLOAD
    )

    fitness_values = np.asarray(stored["fitness"])
    assert fitness_values.size == 1, (
        f"{npz_path}['fitness'] must be the scalar fitness of the stored genome, got "
        f"shape {fitness_values.shape}"
    )
    recorded = float(fitness_values.reshape(-1)[0])
    assert np.isfinite(recorded), (
        f"{npz_path} records the fitness {recorded}; a best genome is selected by a "
        "finite fitness, so the checkpoint cannot hold an invalid one"
    )

    rescored = evaluate_population(
        pipeline, state, _one_individual_payload(state, arrays)
    )
    assert rescored.shape == (1,), (
        f"re-scoring one loaded genome must yield a (1,) fitness array, got "
        f"{rescored.shape}"
    )
    np.testing.assert_allclose(
        float(rescored[0]),
        recorded,
        rtol=1e-5,
        atol=1e-6,
        err_msg=(
            f"the best genome loaded from {npz_path} must re-score to the fitness the "
            "checkpoint recorded; a mismatch means the file does not round-trip the "
            "genome the run selected"
        ),
    )

    history = fitness_history(pipeline)
    assert history, (
        "the run recorded no per-generation fitness, so the checkpoint's fitness "
        "cannot be checked against the log it claims to summarise"
    )
    np.testing.assert_allclose(
        recorded,
        max(best for _, best in history),
        rtol=1e-6,
        atol=1e-6,
        err_msg=(
            f"{npz_path}['fitness'] must be the best fitness the run logged; the "
            f"checkpoint and the log disagree (history maxima: "
            f"{[best for _, best in history]})"
        ),
    )
