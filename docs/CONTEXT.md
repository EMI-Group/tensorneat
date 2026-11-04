# docs

## Intent
Sphinx (autodoc + napoleon + viewcode, theme `sphinx_rtd_theme`) site for the `tensorneat` package; it documents the source tree, it does not replace the tutorials.
`uv run --extra docs sphinx-build -b html docs docs/_build` is the build; `docs/_build/` is gitignored output.

## API Surface
`docs/conf.py` puts `../src` on `sys.path` and keeps `autodoc_default_options = {members, show-inheritance, undoc-members: False}`; it holds no documentation content.
`docs/index.rst` is the front page: intent, quickstart code block, and an "API reference" toctree that points at `source/modules`.
`docs/source/modules.rst` is the API root; it is the only page `index.rst` links to and it toctrees the package root `tensorneat`.

## Constraints
Every page under `docs/source/` must stay reachable from `source/modules.rst` (an unreachable page warns `toc.not_included`, which `-W` turns into a failure).
Package pages (`source/tensorneat.rst`, every `source/tensorneat.<pkg>.rst` for a package directory) are pure landing pages: one sentence of intent plus a `:maxdepth: 4` toctree over that package's subpackages and submodules; they never carry an `automodule`.
Module pages (`source/tensorneat.<module>.rst`) carry exactly one `.. automodule::` with `:members:` and `:show-inheritance:` — that page is the single place an object is documented, because every package `__init__.py` only re-exports and a second registration makes every unqualified cross-reference ambiguous.
When a class documents a member in its own `Attributes:` section, napoleon renders that as an `.. attribute::`, so autodoc's member entry for the same member duplicates the object; the module page excludes that name with `:exclude-members:` instead of hiding the object.
`:exclude-members:` is the only suppression in this tree: `suppress_warnings`, `nitpicky = False` and page-level `:no-index:` are not used.
The remaining build warnings come from docstrings in `src/` (`genome/gene/node/base.py`, `genome/operations/mutation/degree.py`) and must be fixed there, not here.
Pages stay 5–20 lines; a page that needs more is documenting more than one thing.

## Routing Table
| Path | Purpose |
|---|---|
| `./conf.py` | Sphinx configuration only (paths, extensions, autodoc defaults, theme). |
| `./index.rst` | Front page: intent, quickstart, link into the API reference. |
| `./source/modules.rst` | API root toctree. |
| `./source/*.rst` | One page per module (autodoc) or per package (landing page). |
