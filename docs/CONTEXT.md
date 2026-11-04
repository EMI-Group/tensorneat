# docs

## Intent
Sphinx site (autodoc + napoleon + viewcode, theme `sphinx_rtd_theme`) that renders the API of the `src/tensorneat` 0.2.0 tree; it documents the code, it does not replace the `tutorials/` notebooks or the top-level `README.md`.
Build command: `uv run --extra docs sphinx-build -W -b html docs docs/_build` (run `uv sync --extra docs` first; the `docs` extra in `pyproject.toml` is the source of truth).
`docs/requirements.txt` pins only the Sphinx toolchain (`sphinx >= 7.0`, `sphinx-rtd-theme`) and is not a build path: pip cannot install the package from it, because `evoxc >= 0.1.1` only resolves through `[tool.uv.sources]` and PyPI has no satisfying release.
`docs/_build/` is gitignored (`docs/_build/*` in `.gitignore`) and is never committed.
The CI gate is the `docs` job in `.github/workflows/ci.yaml`: `uv sync --extra docs` then the same `-W` build, so a warning fails the pipeline.
The hosted build is driven by the repo-root `.readthedocs.yaml`: ubuntu-24.04 + Python 3.12 (the `requires-python` floor is 3.11), `uv venv $READTHEDOCS_VIRTUALENV_PATH` then `uv sync --extra docs` into that venv — a plain `pip install .` cannot work because `evoxc >= 0.1.1` is resolved from a git tag whose metadata still says `0.1.0`.
RTD builds strictly: `sphinx.fail_on_warning: true`, and because the overridden `build.html` command replaces RTD's default (the place RTD injects its own `-W`), that command passes `-W` explicitly alongside `-T` (full traceback on exceptions, unrelated to warnings).

## API Surface
`docs/conf.py` holds no content: it puts `../src` (absolute, CWD-independent) on `sys.path` when that directory exists, reads `version`/`release` from `importlib.metadata.version("tensorneat")` with a `"0.2.0"` fallback, sets `copyright = "2026, Lishuang Wang"`, `autodoc_default_options = {members, show-inheritance, undoc-members: False}`, `extensions = [autodoc, napoleon, viewcode]`, `exclude_patterns = ["_build", ...]`, `html_theme = "sphinx_rtd_theme"`.
`docs/index.rst` is the front page: intent paragraph (NEAT/HyperNEAT tensorization on `evoxc`, no JAX/PyTorch), a backend paragraph (`numpy` default reference interpreter, `iree` compiled and working end to end via `uv sync --extra iree`, `xla` requires a user-supplied PJRT plugin `.so` via `EVOXC_PJRT_PLUGIN`, `tvm` untested here), a quickstart matching the real `Pipeline`/`algorithm.NEAT`/`problem.XOR` API (`setup()` builds nothing; the per-shape graph is compiled by the first `step`/`auto_run`), an "API reference" toctree pointing at `source/modules`, and GitHub links to `README.md`, `benchmarks/RESULTS.md`, `benchmarks/BACKENDS.md`.
`docs/source/modules.rst` is the API root and the only page `index.rst` links to; it toctrees the package root `tensorneat`.
`docs/source/tensorneat.problem.rl.*.rst` covers the three RL modules (`cartpole` → `CartPole`, `mountaincar` → `MountainCarContinuous`, `pendulum` → `Pendulum`), all three listed by the `tensorneat.problem.rl` package page.
`src/tensorneat/problem/rl/_common.py` is `_`-prefixed and therefore deliberately has no page.

## Constraints
The stub set mirrors `src/tensorneat`: one page per public module (`source/tensorneat.<module>.rst`) and one landing page per package/subpackage (`source/tensorneat.<pkg>.rst`); `_`-prefixed modules (`algorithm/neat/_kernels.py`, `_distance.py`, `_validation.py`) are private and get no page.
When `src/tensorneat` gains or loses a public module, add or delete the matching stub and its toctree entry in the enclosing package page — `find src/tensorneat -name '*.py' | sort` versus `ls docs/source/*.rst` is the sync check.
Module pages carry exactly one `.. automodule::` with `:members:` and `:show-inheritance:`; that page is the single place an object is documented, because every package `__init__.py` merely re-exports and a second registration makes each unqualified cross-reference ambiguous (and emits duplicate-object warnings).
Package pages are pure landing pages: one sentence of intent plus a `:maxdepth: 4` toctree over subpackages/submodules, never an `automodule`.
When a class documents a member in its own `Attributes:` section, napoleon already emits an `.. attribute::` for it, so the module page excludes that name via `:exclude-members:` rather than hiding the object.
`:exclude-members:` is the only suppression used: `suppress_warnings`, `nitpicky = False` and page-level `:no-index:` are not acceptable shortcuts.
Every page must stay reachable from `source/modules.rst`; an unreachable page warns `toc.not_included`, which `-W` turns into a failure.
Pages stay 5–20 lines; a page that needs more is documenting more than one thing.

## Known Issues
None.
`rm -rf docs/_build && uv run --extra docs sphinx-build -W -b html docs docs/_build` exits 0, reports zero warnings, and `-w <log>` leaves that log empty.

## Test Strategy
Verification is the build itself: `rm -rf docs/_build && uv run --extra docs sphinx-build -W -b html docs docs/_build -w /tmp/w.log` must exit 0 with an empty `/tmp/w.log` (0 lines) and print `build succeeded.`.
Spot-check the rendered API: `grep -o 'id="tensorneat.pipeline.Pipeline"' docs/_build/source/tensorneat.pipeline.html` and confirm `docs/_build/index.html` shows the quickstart without leaking literal RST markup.
Spot-check the RL pages the same way: `id="tensorneat.problem.rl.pendulum.Pendulum"`, `id="tensorneat.problem.rl.mountaincar.MountainCarContinuous"`, `id="tensorneat.problem.rl.cartpole.CartPole"`, and all three modules listed by `docs/_build/source/tensorneat.problem.rl.html`.
`uv sync --extra docs` must be run first; `uv` lives at `~/.local/bin/uv` and may not be on `PATH`.

## Routing Table
| Path | Purpose |
|---|---|
| `./conf.py` | Sphinx configuration only (paths, version, extensions, autodoc defaults, theme). |
| `./index.rst` | Front page: intent, quickstart, API-reference toctree, external links. |
| `./requirements.txt` | Sphinx toolchain pin only (`sphinx`, `sphinx-rtd-theme`); not a build path — install the project with `uv sync --extra docs`, never with pip. |
| `./source/` | Autodoc stubs mirroring the `src/tensorneat` tree: `modules.rst` root, package landing pages, one page per module. |
