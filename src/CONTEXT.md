# src — installable source root

## Intent
Setuptools src-layout root: the only importable code in the repository lives under this directory, so packaging (`[tool.setuptools.packages.find] where = ["src"]`) cannot accidentally ship benchmark or test code.

## API Surface
Nothing is importable from `src` itself; `src/tensorneat` is the distribution.

## Constraints
- One top-level package (`tensorneat`); no secondary distribution lives here.
- Nothing under `src/` may import from `benchmarks/`, `test/`, `examples/`, or `tutorials/`.

## Routing Table
| Path | Responsibility |
|---|---|
| `./tensorneat` | The TensorNEAT package: kernel contract, pipeline compile site, genome/algorithm/problem/common subtrees. |
