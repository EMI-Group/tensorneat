# .github/workflows — CI and publishing
## Intent
Hold every GitHub Actions workflow for this repo: `ci.yaml` is the only workflow that runs tests, `pypi_publish.yaml` only publishes a release.
`ci.yaml` is the correctness gate; the performance contract lives in `benchmarks/RESULTS.md` and is not measured in CI (timings on shared runners are not comparable).
## Workflows
- `ci.yaml` — runs on `push` to `main`/`master`, on every `pull_request`, and on `workflow_dispatch`; `permissions: contents: read`; a `concurrency` group cancels superseded runs of the same ref.
- `pypi_publish.yaml` — release-only: on `release: created` it builds an sdist/wheel and uploads via twine with `PYPI_USERNAME`/`PYPI_PASSWORD` secrets; it runs no tests and must stay untouched.
## Jobs In `ci.yaml`
- `test` — the gate: ubuntu-latest + Python 3.12 only, `uv sync --frozen --extra bench` (jax is needed by `test/parity` as the frozen oracle), then `uv run --frozen pytest` (unit, integration, `test/backends` skipping without the `iree` extra, and `test/release_gate_no_jax.py`) and a separate `uv run --frozen pytest -m parity` step.
- `import-hygiene` — base install only (`uv sync --frozen`, no extras), then a one-line `python -c` that fails if any of `jax`/`torch`/`brax`/`gymnax`/`mujoco` entered `sys.modules` during `import tensorneat`.
- `iree` — optional compiled-backend leg: `uv sync --frozen --extra iree` then `uv run --frozen pytest -m requires_iree`; deliberately `continue-on-error: true` and not a dependency of `test`, because the extra pulls an ~80 MB compiler wheel and `iree-compile` costs ~9–11 s per static shape.
## Rules
- `uv sync` always passes `--frozen`: the committed `uv.lock` is authoritative and CI must never re-resolve or rewrite it.
- Python is pinned to `3.12` (the repo's `.python-version`) via `astral-sh/setup-uv@v5` plus an explicit `uv python install 3.12`.
- Action majors are pinned (`actions/checkout@v4`, `astral-sh/setup-uv@v5`); every job sets `timeout-minutes`.
- No Windows runners and no second OS in the matrix: the suite is pure numpy, so extra OS coverage buys little signal for double the cost.
- The `iree` leg must never become a requirement for the `test` verdict; a compiled-backend failure is signal, not a gate.
