# results — generated run records
## Intent
Hold the machine-readable output of `../run_benchmark.py` (`latest.json`: per-run `RunResult` records plus aggregated rows and the protocol/environment snapshot) so the committed `../RESULTS.md` can be re-rendered without re-running the benchmark.
Nothing here is authored by hand and everything here is git-ignored; deleting the directory contents costs nothing but a re-run.
## API Surface
- `latest.json` — written by `run_benchmark.py --json` (default path); schema `tensorneat-bench/1`, documented in `../harness/CONTEXT.md`.
## Constraints
- Never edit a JSON record to make a report look better; regenerate by running the benchmark.
- Keep this directory free of caches and venvs — it is evidence output only.
## Routing Table
| Path | Purpose |
|---|---|
| `../run_benchmark.py` | Producer of `latest.json` and of `../RESULTS.md`. |
| `../harness/reporter.py` | Renderer that turns this JSON into the markdown report. |
