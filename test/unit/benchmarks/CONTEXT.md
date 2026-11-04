# ./benchmarks — regression tests for the benchmark harness fingerprint
## Intent
Pin the one host-side piece of `../../benchmarks/harness` that can silently lie: `protocol.git_report()`, whose `git_dirty` flag is what the report banner calls "re-probed on this tree".
The bug these tests exist for: the probe parsed `git status --porcelain` after stripping the output, so the leading space of `" M path"` was lost, the two-character XY status field was read one character short and the path was sliced from the wrong offset — a modified-but-unstaged file reported `git_dirty: False`.
## What Is Tested
- `git_report()` against a real temporary git repository (`tmp_path`): clean tree → clean, unstaged modification → dirty and names `src/mod.py`, untracked file → dirty, staged rename → dirty and names the new path (never the rename source).
- `_porcelain_path()` on the documented `XY PATH` records (`" M path"`, `"M  path"`, `"?? path"`, `"A  path"`, `"R  old -> new"`, a path with a space) and on truncated records, which yield `""` rather than a shifted fragment.
- `_porcelain_paths()` on `git status --porcelain -z` output: NUL-separated, unquoted, rename destination first with its source in a separate field.
- `_git(..., strip=False)` keeps the first character of the status field; `git_report()` keeps its three public keys and their types.
## Rules
- `../../benchmarks` is a read-only input: the harness is imported as `benchmarks.harness.protocol` (the checkout root is appended to `sys.path` by `../../conftest.py`), and no test writes outside its `tmp_path` repository.
- The repository fixture uses a local git identity and commits once, so every case starts from a genuinely clean tree; a missing `git` binary skips the module instead of asserting something weaker.
- Never soften the assertion: the dirty flag is a claim about whether a measurement is reproducible, so the tests check the parsed path itself, not just the boolean.
