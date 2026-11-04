# legacy_jax — archive of the pre-0.2.0 JAX tutorial notebooks

## Intent
Archive of the three JAX-era teaching notebooks, kept for **concepts only**: topology tensorization, the genome/state layout, and the NEAT usage walkthrough.
None of the code in them runs against the current library: the compute layer is the EvoX Compiler (`evoxc`), `jax` is not a dependency, and `test/release_gate_no_jax.py` forbids importing it.
Read them for the ideas, then follow `../quickstart.ipynb` for anything runnable.

## Contents
- `tutorial-00-functional programming and state.ipynb` — why the library is functional, and the immutable `State` container (`register` / `update` / `save` / `load`).
- `tutorial-01-genome.ipynb` — genome anatomy, batched forward via `jax.vmap`, `genome.execute_mutation`, `network_dict` plus SVG drawing.
- `tutorial-02-Usage.ipynb` — the problem/algorithm/pipeline walkthrough: a custom jitable problem, a hand-written `ask`/`tell` loop, and an "un-jitable" CartPole chapter built on `gymnasium`.
- `origin_network.svg`, `mutated_network.svg` — the two figures `tutorial-01` writes and embeds.
- `tutorial_0_state.pkl` — the pickle `tutorial-00` writes: `State(a=1, b=2, c=3, d=4)`, a plain key/value dict with no tensors, so the current `State.load` still reads it.

## Where each old concept lives now
| Old notebook | Old API (does not exist) | Current equivalent |
|---|---|---|
| `tutorial-00` | `@jax.jit` around a function that takes/returns `State`, `jax.numpy` arrays | `@evoxc.defn` graph + `evoxc.build` — see `src/tensorneat/pipeline.py` and `src/tensorneat/common/CONTEXT.md`; batch axes are written explicitly, `vmap` is never applied |
| `tutorial-00` | `State(...)`, `.register`, `.update`, `.save`, `.load` | same names in `src/tensorneat/common/state.py`, now a host-only immutable container that never enters a trace; randomness lives in `evoxc.random` keys on the state |
| `tutorial-01` | `genome.setup()`, `genome.initialize(state, jax.random.PRNGKey)` | `genome.setup(state=None) -> State` and `evoxc.random.key(seed)` — see `src/tensorneat/genome/` |
| `tutorial-01` | `genome.transform(state, nodes, conns)` + `jax.vmap(genome.forward)` | `transform(state, *arrays) -> Transformed` (the execution schedule) and a batch-first (axis-0) `forward` — see `src/tensorneat/genome/base.py` and `src/tensorneat/common/tensorization.py` |
| `tutorial-01` | `genome.network_dict(state, nodes, conns)`, `genome.visualize(network, save_path=...)` | `genome.network_dict(state, best)` on a **one-individual** payload, then `genome.visualize(network)` — see `examples/interpret_visualize/visualize_genome.py` |
| `tutorial-02` | `BaseProblem.jitable = True` and `evaluate` using `jax.vmap(act_func)` | `problem.BaseProblem.evaluate(state, key, act_fn, transformed)` with the dataset tiled inside the graph; the `jitable` flag is gone — see `src/tensorneat/problem/CONTEXT.md` |
| `tutorial-02` | `jax.jit(algorithm.ask)` / `jax.jit(algorithm.tell)` driven by a Python loop | `Pipeline.step(state)` (one graph run per generation) and `Pipeline.auto_run(state)` |
| `tutorial-02` | the "un-jitable problem" chapter: `gymnasium` CartPole stepped from Python | in-graph `problem` CartPole rollout (`src/tensorneat/problem/rl/`, `examples/rl/cartpole.py`); there is no eager/JIT boundary to cross |
| `tutorial-02` | the end-to-end usage walkthrough | `../quickstart.ipynb`, plus the runnable scripts under `examples/` |

## Constraints
- Do not execute these notebooks, and do not copy their imports or idioms into current material.
- Keep the notebooks and assets byte-identical: this is an archive, so corrections belong in `../quickstart.ipynb` or the module `CONTEXT.md` the concept maps to.
- The `.pkl` and the SVGs exist only to make the archived notebooks self-explanatory; nothing in `src/`, `test/` or `examples/` reads them.

## Routing Table
| Path | Purpose |
|---|---|
| `./tutorial-00-functional programming and state.ipynb` | Archived: functional style and the `State` container (JAX-era). |
| `./tutorial-01-genome.ipynb` | Archived: genome operations, batched forward, mutation, network drawing (JAX-era). |
| `./tutorial-02-Usage.ipynb` | Archived: problem/algorithm/pipeline walkthrough, custom and "un-jitable" problems (JAX-era). |
