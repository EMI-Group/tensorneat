# docs

## Intent
Sphinx site (autodoc + napoleon + viewcode, theme `sphinx_rtd_theme`) that renders the API of the `src/tensorneat` 0.2.0 tree; it documents the code, it does not replace the `tutorials/` notebooks or the top-level `README.md`.
Build command: `uv run --extra docs sphinx-build -b html docs docs/_build` (the `docs` extra is declared in `pyproject.toml`; `docs/requirements.txt` mirrors it for a plain `pip install -r` build).
`docs/_build/` is gitignored (`docs/_build/*` in `.gitignore`) and is never committed.
The hosted build is driven by the repo-root `.readthedocs.yaml`: ubuntu-24.04 + Python 3.12 (the `requires-python` floor is 3.11), `uv venv $READTHEDOCS_VIRTUALENV_PATH` then `uv sync --extra docs` into that venv — a plain `pip install .` cannot work because `evoxc >= 0.1.1` is resolved from a git tag whose metadata still says `0.1.0`.

## API Surface
`docs/conf.py` holds no content: it puts `../src` (absolute, CWD-independent) on `sys.path` when that directory exists, reads `version`/`release` from `importlib.metadata.version("tensorneat")` with a `"0.2.0"` fallback, sets `autodoc_default_options = {members, show-inheritance, undoc-members: False}`, `extensions = [autodoc, napoleon, viewcode]`, `exclude_patterns = ["_build", ...]`, `html_theme = "sphinx_rtd_theme"`.
`docs/index.rst` is the front page: intent paragraph (NEAT/HyperNEAT tensorization on `evoxc`, no JAX/PyTorch), a quickstart matching the real `Pipeline`/`algorithm.NEAT`/`problem.XOR` API, an "API reference" toctree pointing at `source/modules`, and GitHub links to `README.md`, `benchmarks/RESULTS.md`, `benchmarks/BACKENDS.md`.
`docs/source/modules.rst` is the API root and the only page `index.rst` links to; it toctrees the package root `tensorneat`.

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
`-W` (warnings-as-errors) still fails on 3 docutils warnings that originate in `src/` docstrings, not in `docs/`: `genome/gene/node/base.py` (`BaseNode.to_dict`, unterminated inline literal after napoleon splits the `Returns:` block) and `genome/operations/mutation/degree.py` (module docstring: "Unexpected indentation" plus "Block quote ends without a blank line").
A plain (non-`-W`) build therefore exits 0 with exactly those 3 warnings and no others; they must be fixed in `src/`, not suppressed here.

## Test Strategy
Verification is the build itself: `rm -rf docs/_build && uv run --extra docs sphinx-build -b html docs docs/_build -w /tmp/w.log`, then assert `/tmp/w.log` contains only the 3 `src/`-owned docutils lines above.
Spot-check the rendered API: `grep -o 'id="tensorneat.pipeline.Pipeline"' docs/_build/source/tensorneat.pipeline.html` and confirm `docs/_build/index.html` shows the quickstart without leaking literal RST markup.
`uv sync --extra docs` must be run first; `uv` lives at `~/.local/bin/uv` and may not be on `PATH`.

## Routing Table
| Path | Purpose |
|---|---|
| `./conf.py` | Sphinx configuration only (paths, version, extensions, autodoc defaults, theme). |
| `./index.rst` | Front page: intent, quickstart, API-reference toctree, external links. |
| `./requirements.txt` | `docs`-extra dependencies for a plain `pip install -r` build. |
| `./source/` | Autodoc stubs mirroring the `src/tensorneat` tree: `modules.rst` root, package landing pages, one page per module. |
