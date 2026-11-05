"""Regression tests for ``run_benchmark.py --persist-note``.

``--persist-note`` exists so a ``--report-note`` survives the next bare
``--render-from-json``: it stores the note in the record's ``meta.report_note`` and nothing
else. That is a write into committed evidence, so the contract is narrow and enforced twice —
the CLI re-reads the file and refuses to render if any top-level key other than ``meta`` moved,
and these tests pin it from the outside: every other key, every other ``meta`` key, and the
bytes of the whole document must be what the CLI wrote before, with only that one string
replaced.

It has exactly two supported modes. Outside them it exits ``2`` with a named error and writes
nothing at all: without ``--render-from-json`` (it would be a write on the measurement path)
and without ``--report-note`` (there is nothing to persist).

Everything runs on a ``tmp_path`` copy of the committed record and a ``tmp_path`` report:
``benchmarks/`` is a read-only input of this suite, and a smoke render must never clobber the
committed ``RESULTS.md``.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from benchmarks import run_benchmark

NOTE = "unit-test note: only meta.report_note may change"
SOURCE_RECORD = Path(run_benchmark.__file__).resolve().parent / "results" / "authoritative.json"


@pytest.fixture
def record(tmp_path: Path) -> Path:
    """A copy of the committed authoritative record inside a ``results/`` directory of its own."""
    assert SOURCE_RECORD.is_file(), f"the committed record is missing: {SOURCE_RECORD}"
    directory = tmp_path / "results"
    directory.mkdir()
    copy = directory / "authoritative.json"
    shutil.copyfile(SOURCE_RECORD, copy)
    return copy


def _render_args(record: Path, out: Path) -> list[str]:
    """The CLI argv of one ``--render-from-json`` invocation writing ``out``."""
    return ["--render-from-json", str(record), "--out", str(out)]


def test_persist_note_rewrites_only_meta_report_note(record: Path, tmp_path: Path) -> None:
    """After ``--persist-note`` the record differs in exactly ``meta.report_note``."""
    before_text = record.read_text(encoding="utf-8")
    before = json.loads(before_text)
    # The record is a `_write_json` product, so it re-serialises byte-identically; that is what
    # makes the byte comparison below a statement about --persist-note alone.
    assert json.dumps(before, indent=2) + "\n" == before_text

    out = tmp_path / "a.md"
    code = run_benchmark.main([*_render_args(record, out), "--report-note", NOTE, "--persist-note"])
    assert code == 0

    after_text = record.read_text(encoding="utf-8")
    after = json.loads(after_text)

    assert after["meta"]["report_note"] == NOTE
    assert after["runs"] == before["runs"]
    for key in before:
        if key == "meta":
            continue
        assert after[key] == before[key], f"top-level key {key!r} changed"
    assert set(after) == set(before)
    assert {k: v for k, v in after["meta"].items() if k != "report_note"} == {
        k: v for k, v in before["meta"].items() if k != "report_note"
    }

    expected = {**before, "meta": {**before["meta"], "report_note": NOTE}}
    assert after_text == json.dumps(expected, indent=2) + "\n"

    assert out.is_file()
    assert NOTE in out.read_text(encoding="utf-8")


def test_a_bare_rerender_keeps_the_persisted_note(record: Path, tmp_path: Path) -> None:
    """The note comes from the record, so a later render without ``--report-note`` carries it."""
    first = tmp_path / "a.md"
    assert (
        run_benchmark.main([*_render_args(record, first), "--report-note", NOTE, "--persist-note"])
        == 0
    )
    persisted = record.read_bytes()

    second = tmp_path / "b.md"
    assert run_benchmark.main(_render_args(record, second)) == 0

    assert NOTE in second.read_text(encoding="utf-8")
    # A render never rewrites the record it reads.
    assert record.read_bytes() == persisted


def test_persist_note_without_render_from_json_exits_2_and_writes_nothing(
    record: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """On the measurement path ``--persist-note`` is refused before anything is written."""
    before = record.read_bytes()
    out = tmp_path / "a.md"
    json_out = tmp_path / "a.json"
    # No --render-from-json: this is the measurement path, where the flag must be refused
    # before the plan is even validated (the paths below exist only so a regression that
    # ignored the guard would write into tmp_path and not into the repository).
    code = run_benchmark.main(
        [
            "--cases",
            "xor_p100",
            "--report-note",
            NOTE,
            "--persist-note",
            "--out",
            str(out),
            "--json",
            str(json_out),
        ]
    )
    stderr = capsys.readouterr().err
    assert code == 2
    assert "--persist-note needs --render-from-json" in stderr
    assert not out.exists()
    assert not json_out.exists()
    assert record.read_bytes() == before


def test_persist_note_without_report_note_exits_2_and_writes_nothing(
    record: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """``--persist-note`` with nothing to persist fails loudly instead of writing a null note."""
    before = record.read_bytes()
    out = tmp_path / "a.md"
    code = run_benchmark.main([*_render_args(record, out), "--persist-note"])
    stderr = capsys.readouterr().err
    assert code == 2
    assert "--persist-note without --report-note" in stderr
    assert not out.exists()
    assert record.read_bytes() == before
