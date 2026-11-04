"""Regression tests for the harness fingerprint (``benchmarks/harness/protocol.py``).

``git_report`` feeds the report banner that says the dirty state is re-probed on this tree,
so a misparsed ``git status`` turns a dirty measurement into a claim of a clean one.
Git's short/porcelain format is ``XY PATH``: exactly two status characters, one space, then
the path — which is why a worktree-only change starts with a space (``" M src/mod.py"``).
The harness used to strip the probe output before parsing, losing that leading space on the
first record, and then sliced the path at the wrong offset (``"M src/mod.py"`` -> ``"c/mod.py"``
once the status field lost a character), so a modified-but-unstaged file read as clean.
The fix parses ``--porcelain -z`` (NUL-separated, unquoted, rename destination first) and
never strips the probe output.

These tests pin the parser on the documented records and pin the flag on a real repository
created under ``tmp_path``: clean tree, unstaged modification, untracked file, staged rename.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

from benchmarks.harness import protocol

GIT = shutil.which("git")

pytestmark = pytest.mark.skipif(GIT is None, reason="the fingerprint needs a git binary")


def _run(root: Path, *arguments: str) -> str:
    """Run one ``git`` command inside ``root`` and return its stdout verbatim."""
    completed = subprocess.run(
        [GIT or "git", *arguments],
        cwd=str(root),
        capture_output=True,
        text=True,
        check=True,
    )
    return completed.stdout


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    """A real, committed git repository holding a tiny clean tree with code and evidence."""
    root = tmp_path / "repo"
    (root / "src").mkdir(parents=True)
    (root / "benchmarks" / "results").mkdir(parents=True)
    (root / "src" / "mod.py").write_text("VALUE = 1\n", encoding="utf-8")
    (root / "pyproject.toml").write_text("[project]\nname = 'fixture'\n", encoding="utf-8")
    (root / "benchmarks" / "results" / "latest.json").write_text("{}\n", encoding="utf-8")
    _run(root, "init", "-q", ".")
    _run(root, "config", "user.email", "bench@example.invalid")
    _run(root, "config", "user.name", "bench")
    _run(root, "add", "-A")
    _run(root, "commit", "-q", "-m", "fixture: clean tree")
    return root


def test_clean_tree_reports_clean(repo: Path) -> None:
    """(a) A committed tree with no change is clean, and the public shape is unchanged."""
    report = protocol.git_report(repo)
    assert set(report) == {"git_commit", "git_dirty", "git_dirty_paths"}
    assert report["git_commit"] == _run(repo, "rev-parse", "HEAD").strip()
    assert len(report["git_commit"]) == 40
    assert report["git_dirty"] is False
    assert report["git_dirty_paths"] == []


def test_unstaged_modification_reports_dirty(repo: Path) -> None:
    """(b) The reported bug: ``" M src/mod.py"`` used to parse as ``"c/mod.py"`` and read clean."""
    (repo / "src" / "mod.py").write_text("VALUE = 2\n", encoding="utf-8")
    status = _run(repo, "status", "--porcelain", "--", "src")
    assert status.startswith(" M "), "the leading space is the worktree half of the XY field"
    report = protocol.git_report(repo)
    assert report["git_dirty"] is True
    assert report["git_dirty_paths"] == ["src/mod.py"]


def test_untracked_file_reports_dirty(repo: Path) -> None:
    """(c) An untracked file inside the code scope is dirty."""
    (repo / "src" / "extra.py").write_text("VALUE = 3\n", encoding="utf-8")
    report = protocol.git_report(repo)
    assert report["git_dirty"] is True
    assert report["git_dirty_paths"] == ["src/extra.py"]


def test_staged_rename_reports_dirty_and_names_the_new_path(repo: Path) -> None:
    """(d) A staged rename is dirty; ``-z`` puts the destination first and the source second."""
    _run(repo, "mv", "src/mod.py", "src/renamed.py")
    report = protocol.git_report(repo)
    assert report["git_dirty"] is True
    assert report["git_dirty_paths"] == ["src/renamed.py"]


def test_git_probe_keeps_the_leading_status_character(repo: Path) -> None:
    """``strip=False`` is the other half of the fix: it preserves the XY field's first char."""
    (repo / "src" / "mod.py").write_text("VALUE = 2\n", encoding="utf-8")
    raw = protocol._git("status", "--porcelain", "-z", "--", "src", root=repo, strip=False)
    assert raw is not None
    assert raw.startswith(" M src/mod.py\0")


@pytest.mark.parametrize(
    ("record", "expected"),
    [
        (" M src/mod.py", "src/mod.py"),
        ("M  src/mod.py", "src/mod.py"),
        ("?? src/mod.py", "src/mod.py"),
        ("A  src/added.py", "src/added.py"),
        ("R  src/old.py -> src/mod.py", "src/mod.py"),
        (" M src/a b.py", "src/a b.py"),
    ],
)
def test_porcelain_path_reads_the_two_character_status_field(record: str, expected: str) -> None:
    """Every record starts ``XY `` (two status characters plus one space), then the path."""
    assert protocol._porcelain_path(record) == expected


@pytest.mark.parametrize("record", ["", "??", " M "])
def test_porcelain_path_reports_no_path_for_a_truncated_record(record: str) -> None:
    """A record without a path payload yields ``""`` instead of a shifted fragment."""
    assert protocol._porcelain_path(record) == ""


def test_porcelain_paths_splits_nul_records_and_skips_the_rename_source() -> None:
    """``-z`` records are NUL-separated; a rename/copy carries its source in a second field."""
    output = " M src/a.py\0?? src/b c.py\0R  src/new.py\0src/old.py\0"
    assert protocol._porcelain_paths(output) == ["src/a.py", "src/b c.py", "src/new.py"]


def test_porcelain_paths_of_a_clean_tree_is_empty() -> None:
    """No record, no path."""
    assert protocol._porcelain_paths("") == []
    assert protocol._porcelain_paths("\0") == []
