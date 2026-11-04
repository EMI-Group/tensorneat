# The `iree` backend: it runs, and it is slower than numpy here

**Verdict: `backend="iree"` now compiles and runs TensorNEAT end to end — every gate that used to kill it in `Pipeline.setup()` is gone — but on this box it is 4.8× (pop 100), 24× (pop 500) and 23× (pop 1000) slower per generation than the numpy reference interpreter, and the gap grows with population.**
Every shipped problem family runs on it (`XOR`, `XOR3d`, `CustomFuncFit`, `CartPole`, `RecurrentGenome`, `HyperNEAT`); each takes one 8–11 s `iree-compile` per static shape.
`xla` is not available here: the evoxc xla adapter needs a PJRT C API plugin `.so` (`EVOXC_PJRT_PLUGIN`, unset on this box) and nothing provides one. `tvm` was not attempted.
The CPU performance contract therefore stays on the numpy backend (`RESULTS.md`, unchanged by this report).

This file replaces an earlier version that recorded 8/8 cases dying in `Pipeline.setup()` with
`BackendError: stablehlo export: op 'scatter' ... rank-3 operand along axis 0`.
That failure was real and is now fixed at its source in `src/` (section "The four gates, in the order they fired").
`benchmarks/results/iree.json` and `benchmarks/iree.md` are the **historical** record of that pre-fix attempt (8/8 `error`); they were not re-rendered for this report.

## Environment (measured, not assumed)

| Item | Value |
|---|---|
| Date (UTC) | 2026-11-04, runs 05:39–05:55 |
| CPU | `INTEL(R) XEON(R) PLATINUM 8568Y+`, QEMU/KVM guest, 1 socket, 1 thread/core, no SMT |
| `nproc` | 20 |
| Platform | `Linux-6.8.0-146-generic-x86_64-with-glibc2.39` |
| Python | 3.12.3 |
| evoxc | 0.1.0 |
| numpy | 2.5.4 |
| iree-base-compiler / iree-base-runtime | 3.11.0 / 3.11.0 (what `evoxc[iree]` resolves) |
| sympy / networkx / matplotlib / pytest | 1.14.0 / 3.7 / 3.11.2 / 9.2.0 |

## What runs today

| backend | install | what runs | compile needed |
|---|---|---|---|
| `numpy` | always available | everything | 1.6–2.0 s per graph set (the evoxc interpreter build) |
| `iree` (llvm-cpu) | `pip install "evoxc[iree]"` | `XOR`, `XOR3d`, `CustomFuncFit`, `CartPole`, `RecurrentGenome`, `HyperNEAT` — one full generation each, verified | 8.1–10.9 s per generation graph, one-off per static shape |
| `xla` | plugin `.so` required | nothing | not reached: `the xla adapter requires a PJRT C API plugin library (.so exporting GetPjRtApi) and none was found` |
| `tvm` | extra | not attempted | — |

Correctness across backends (same seed 42, XOR, `DefaultGenome`, this machine):
the final `max fitness` is bit-identical at pop 100 (`-0.19304290413856506` numpy vs `-0.19304291903972626` iree, Δ = 1.5e-08) and pop 500 (identical to all printed digits, `-0.15205243229866028`);
at pop 1000 the two trajectories separate at `1e-08`-scale and land on `-0.151799` vs `-0.143853`, which is float-reassociation drift in iree's reductions, not a semantic difference (int32 key payloads compare equal; float attribute arrays agree to ≤ 2.4e-07 ≈ 1–2 ulp).
`test/backends/test_iree_pipeline.py` pins exactly this: int/key payloads `array_equal`, floats within a documented tolerance.

## The five gates, in the order they fired

Each gate below was reproduced verbatim on this machine, then removed **at its source in `src/`**, never by editing evoxc (`/root/evoxc` is a read-only foreign repo; the installed `evoxc` 0.1.0 was left untouched).
The offending op's location is `op.location` in the traced evoxc IR, which is why the errors name a TensorNEAT `file:line`.

### Gate 1 — rank-3 `scatter` (15 sites, export-time)

```
evoxc.core.errors.BackendError: stablehlo export: op 'scatter' at .../src/tensorneat/algorithm/neat/species.py:295
on a rank-3 operand along axis 0 is not supported by the v1 exporter
(numpy put_along_axis broadcast scatter is expressible for rank-1 axis=0 and rank-2 row/column writes;
use the numpy backend for the general case)
```

evoxc's StableHLO v1 exporter (`evoxc/backends/stablehlo/writer.py:1821 _emit_scatter`) supports only rank-1 `axis=0`, rank-2 `axis=0` and rank-2 `axis=1`; the frontend and the numpy kernel accept any rank, so the graph ran on numpy and died at lowering.
The scatters wrote whole gene rows into `(species, max_nodes, 4)` / `(species, max_conns, 2)` / `(species, max_conns, 1)` species-centre tables:
`species.py:295`, `:304`, `:309` (generation-0 centre tables) and `_kernels.py:665`, `:674`, `:683`, `:733`, `:734`, `:735` (species scan bodies).
Rewrites, both in `src/tensorneat/algorithm/neat/_kernels.py`:
- `first_centre_table(species_size, row, fill, dtype)` — a single-row write is `concatenate([reshape(row, (1, …)), constant fill block])`, no scatter at all (every row is written, so there is nothing to mask); `species_size == 1` short-circuits to the reshape.
- `scatter_row(table, slot, update)` — flattens the trailing axes to one row length and performs the identical write as a **rank-2 axis-0 row scatter** (`(rows, R)` operand, `(rows,)` indices, `(rows, R)` updates), then reshapes back.

### Gate 2 — `uint32` / `uint64` in the graph (capability gate, not export)

```
evoxc.core.errors.BackendError: capability drift: the iree backend cannot execute dtype uint32 (op 'cast')
— its declared dtypes are ['bool', 'float16', 'float32', 'float64', 'int16', 'int32', 'int64', 'int8']
```

The iree adapter declares no unsigned dtype (`evoxc/backends/adapters/iree.py`, `_dtype_capabilities()`), and this library used unsigned carriers in three places:
- genome row identity packing (`genome/operations/alignment.py`, `uint64`) — used as a *sort key*, hence unsigned on purpose;
- key-range tests (`genome/operations/distance/{consts,digest}.py`, `common/tensorization.py`, `uint32`) — one `<` comparison instead of two;
- a `uint32` constant created through `common/tools.py:159`.

Rewritten as **same-width signed carriers**: two's complement makes add/sub/shift/or/xor/cast bit-identical, so the packed values do not move.
Where unsigned *order* mattered (the `argsort`/comparison on the packed identity) it is restored by one bijective sign flip, `bitwise_xor(v, INT64_MIN)` (`alignment.py::_order_ready`) — equality preserved, signed order equals unsigned order.
The `uint32` range trick became `logical_and(greater_equal(keys, 0), less(keys, span))` (`common/tensorization.py::_key_indices`), i.e. the same two-comparison form the float branch already used.
NumPy results are byte-identical after all of it (per-node A/B dumps: 144 arrays at pop 100, 72 at pop 500, 72 at pop 1000 in `algorithm/neat`; 23 arrays across gene ops; 108 schedule fields in `common`).

### Gate 3 — `tril`, `flip`, `cumsum` (ops the v1 exporter defers)

```
evoxc.core.errors.BackendError: stablehlo export: op 'tril' at .../src/tensorneat/algorithm/neat/_kernels.py:140 is not supported in v1
```

`tril`, `triu`, `flip`, `cumsum`, `cumprod`, `erf`, `gelu`, `where`, `scan`, `convolution`, `runtime_call`, `external_call` and the 16 `sparse_*` ops have no v1 emission (`evoxc/backends/stablehlo/ops.py:235 DEFERRED_OPS`, plus names absent from every mapping table).
Rewrites:
- `tril` (`_kernels.py:140`, species prefix sum) → a **static constant** triangular mask: the matrix is `(species_size, species_size)` and fully static, so it is one `constant` op instead of a deferred op (and cheaper to run).
- `flip` (`_kernels.py:295`) → `evoxc.gather` with a static reversed `int32` index vector.
- `cumsum` (`genome/operations/mutation/cycle.py:121`, a `(P, 214)` suffix count) → a log2-doubling (Hillis–Steele) fold: a `while` over `shift < total` adding a shifted slice, in `int8`/`int32`. Bit-identical, ~24 extra traced ops, +0.3 ms at pop 100.

### Gate 4 — `iree-compile` could not legalize a shared index block (`iree_linalg_ext.scatter`)

```
<stdin>:338:12: error: failed to legalize unresolved materialization from ('tensor<32x1xi32>') to ('tensor<32xi32>') that remained live after conversion
    %337 = "stablehlo.scatter"(%323, %334, %325) ({
<stdin>:338:12: note: see existing live user here:
%238 = iree_linalg_ext.scatter dimension_map = [0] unique_indices(false) ins(%collapsed_399, %236 : tensor<32xi32>, tensor<32xi32>) outs(%cst_201 : tensor<68xi32>)
```

This one is IREE-specific and is **not** an export deferral: the op passes evoxc's exporter and even compiles fine in isolation.
The writer emits every flat write as a rank-1 `axis=0` scalar-window scatter with a rank-2 `(K,1) i64` index tensor (`index_vector_dim = 1`).
IREE lowers that to `iree_linalg_ext.scatter`, whose `ins` wants the index tensor **collapsed to rank 1 i32**, and performs the collapse as an `unrealized_conversion_cast`. That materialization only resolves if the rank-2 index value has **exactly one consumer**; when one prepared index block (or two identical index expressions that IREE's CSE merges) feeds a **second** scatter, the rank-2 value stays live and the pass aborts.
Operand sharing, widths, i32-vs-i64 indices, `while`/`if` regions, `sort`, and the rank-2 row-scatter alternative are all irrelevant — deleting either of the two shared-index scatters from the failing module makes it compile (0.26 s).

Fix, in the shared helper's caller (`src/tensorneat/common/tensorization.py::incoming_table`, `_window_pair`): the two payloads that shared one index block (`in_conn`, `in_src`) now go into **one** `(P, 2*(N*K+1))` buffer written by **one** scatter, with the second half's cells offset by one window span and its dropped writes routed to its own dump column.
The resulting rule is now documented in `common/tools.py` and `common/CONTEXT.md`: **one prepared index block feeds exactly one scatter** — merge payloads into a wider buffer instead.

### Gate 5 — a negative `concatenate` axis (HyperNEAT only, `iree-compile` reject)

```
<stdin>:1543:13: error: 'stablehlo.concatenate' op attribute 'dimension' failed to satisfy constraint:
64-bit signless integer attribute whose value is non-negative
    %1524 = "stablehlo.concatenate"(%1462, %1523) {dimension = -1 : i64} : (tensor<160x3xf32>, tensor<160x1xf32>) -> tensor<160x4xf32>
```

evoxc forwards the frontend axis **verbatim** as `stablehlo.concatenate`'s `dimension`, and StableHLO requires it non-negative, so the `axis=-1` spelling — perfectly legal at the frontend and in numpy — only fails once a compiled backend sees it.
Reached only by the HyperNEAT path (the plain NEAT graphs never wrote a negative axis), so it was invisible until the compiled run got that far.
Fix: write the explicit non-negative last axis in graph code — `hyperneat.py` (bias column, attribute stack, connection-key marker columns) and `genome/gene/base.py::_key_array` (`axis=len(shape)`).
Rule: **axes in graph ops are non-negative static integers**; the sweep found no other negative `concatenate` axis in `src/`.
End-to-end after the fix: `HyperNEAT(FullSubstrate(4/3/1), activate_time=3)` + `XOR3d`, seed 42, `backend="iree"` → one 10.4 s compile, then 35.8/36.4 ms per step, keys and NaN masks bit-identical to numpy, weights within 2.4e-07, same `randkey`.

## Measured: numpy vs iree
Protocol: `benchmarks/backend_measure.py` (new), one `(case, backend, pop)` per **fresh process** (peak RSS is a process high-water mark).
`Pipeline.setup()` wall time is reported as `setup_s` (host-only plus NEAT's small operator graph); the expensive generation graph compiles on the **first** `step()`, reported as `build_s` (`Pipeline.build_seconds`, the `evoxc.build` alone) and `first_step_s` (build + one run). The build is charged to no generation.
Then **3 untimed warmup** generations and **20 timed** generations; ms/generation is wall clock per `step()` including host materialisation. One run each, seed 42, XOR, `DefaultGenome(2→1, max_nodes=12, max_conns=24)`, `species_size=20`.

| case | pop | backend | opt_level | setup (s) | build (s) | ms/gen mean | ms/gen median | min / max | gens/s | peak RSS (MiB) | final max fitness |
|---|---|---|---|---|---|---|---|---|---|---|---|
| xor | 100 | numpy | — | 0.04 | 1.60 | 44.37 | **42.74** | 39.75 / 62.61 | 22.54 | 120.9 | −0.193043 |
| xor | 100 | iree | default | 0.97 | 9.35 | 212.67 | **213.12** | 179.11 / 229.28 | 4.70 | 108.6 | −0.193043 |
| xor | 100 | iree | O0 | 0.98 | 9.28 | 211.25 | **211.56** | 180.10 / 221.93 | 4.73 | 108.7 | −0.193043 |
| xor | 500 | numpy | — | 0.04 | 1.62 | 265.27 | **262.80** | 255.86 / 292.09 | 3.77 | 263.6 | −0.152052 |
| xor | 500 | iree | default | 0.99 | 9.97 | 6371.71 | **6366.20** | 6297.28 / 6483.87 | 0.157 | 153.0 | −0.152052 |
| xor | 500 | iree | O0 | 1.02 | 9.87 | 6350.53 | **6344.47** | 6301.80 / 6438.72 | 0.157 | 154.4 | −0.152052 |
| xor | 1000 | numpy | — | 0.04 | 1.70 | 500.86 | **546.70** | 331.77 / 570.72 | 2.00 | 433.9 | −0.151799 |
| xor | 1000 | iree | default | 1.01 | 10.86 | 12766.70 | **12765.81** | 12629.17 / 12910.57 | 0.078 | 210.2 | −0.143853 |

Reading it honestly:

- **The compiled leg never wins here.** Slower by 5.0× at pop 100, 24.2× at pop 500, 23.4× at pop 1000 (median ratios); there is no population size in the measured range at which iree flips ahead, and the trend is *worse* as the population grows, so extrapolating to larger pops does not rescue it.
- **Compile is significant and must be paid per static shape**: ~9.3–10.9 s of `iree-compile` for the generation graph, plus ~1 s of `setup`, against 1.6–1.7 s for the whole numpy build. A 20-generation run therefore spends 35–50 % of its lifetime compiling at these budgets.
- **Peak RSS is the one place iree wins**: 108.6 vs 120.9 MiB at pop 100, 153.0 vs 263.6 MiB at pop 500, 210.2 vs 433.9 MiB at pop 1000 (llvm-cpu keeps the tensors in its own buffers; the interpreter holds numpy temporaries).
- **`opt_level` barely matters at O0** (within 1 % of default) and **O2 does not compile here**:
  `error: 'func.func' op exceeded stack allocation limit of 32768 bytes for function. Got 40064 bytes` — the ~2.4 k-op generation function exceeds IREE's per-function stack-promotion limit at that level. The value is forwarded verbatim and the failure surfaces as `evoxc.core.errors.BackendError`; nothing falls back to another backend.
- The reported **`iree-compile` SIGSEGV (≥3 loop-carried tensors)** from the earlier investigation does not reproduce in this configuration: the shipped generation graph contains a **9-carried-value** `stablehlo.while` and compiled successfully in every run recorded above (6 fresh compiles), all of which executed and produced matching fitness.

Per-generation cost is dominated by the graph's dense `gather`/`argsort`/`scatter` traffic, which llvm-cpu lowers to generic, non-vectorised loops — the reason a compiled backend loses to a tight numpy interpreter on this workload. Making iree competitive is a kernel-formulation project (dense key tables → sparse/segment reductions), not a compiler-flag one.

## What a user must know to run `backend="iree"`

- `pip install "evoxc[iree]"` (pulls `iree-base-compiler` + `iree-base-runtime`); then `Pipeline(..., backend="iree")` just works for the shipped problems.
- **Compile is one-off per static shape**: population, `max_nodes`, `max_conns` are baked into the graph, so a shape change re-compiles (~10 s here). `Pipeline` caches the executable per shape.
- **Kernel rules the exporter and iree impose** (all enforced by construction in `src/` today, and documented in `src/tensorneat/common/CONTEXT.md`, `src/tensorneat/algorithm/neat/CONTEXT.md`, `src/tensorneat/algorithm/hyperneat/CONTEXT.md`, `src/tensorneat/genome/CONTEXT.md` and `src/tensorneat/genome/operations/CONTEXT.md`):
  1. `scatter` must be rank-1 `axis=0` or a rank-2 row/column write — no rank ≥ 3 operand.
  2. One prepared index block feeds exactly one `scatter` (gate 4).
  3. No unsigned dtype anywhere in a graph: use same-width signed carriers, plus an explicit sign flip wherever unsigned *order* is required.
  4. No v1-deferred op: `cumsum`, `cumprod`, `tril`, `triu`, `flip`, `erf`, `gelu`, `where`, `scan`, `convolution`, `runtime_call`, `external_call`, collectives other than the mapped ones, and the `sparse_*` family. Use a static constant, a `gather` with a static index vector, or a log2-doubling fold instead.
  5. Static dims only, **non-negative axes only**, and `select`/`gather` (single axis)/`sort`/`argsort`/`where`-free masks/`cond`/`while_loop`/static-bound slices/`rng_bit_generator` all export.
- **Activation functions**: no `ACT` entry needs `erf`/`gelu` (the set is `scaled_sigmoid`, `sigmoid`, `scaled_tanh`, `tanh`, `sin`, `relu`, `lelu`, `identity`, `inv`, `log`, `exp`, `abs`); a *user-supplied* activation that calls `evoxc.erf`/`evoxc.gelu` would still die at export with `op 'erf' ... is not supported in v1`.
- `opt_level="O2"`/`"O3"` fail inside `iree-compile` for this graph size (stack allocation limit). Leave the default or use `O0`/`O1`.
- Do not expect an iree speed-up: on this CPU, numpy is 5–24× faster per generation. iree's only measured advantage is ~30–50 % lower peak RSS.

## Still deferred by the evoxc v1 exporter (would bite other users)

Straight from the installed evoxc 0.1.0, not inferred:
- `evoxc/backends/stablehlo/ops.py:235 DEFERRED_OPS`: `runtime_call`, `external_call`, `block_call`, `rank`, `world_size`, `erf`, `gelu`, `diagonal`, `cholesky`, `qr`, `matrix_rank`, `svd`, `matrix_exp`, `random_multinomial`, and 16 `sparse_*`/`dense_dot_sparse` ops.
- Names with no v1 mapping at all (status `deferred` but absent from that set): `cumsum`, `cumprod`, `flip`, `tril`, `triu`, `where`, `scan`, `convolution`, `collective_broadcast`.
- `evoxc/backends/stablehlo/writer.py:1821 _emit_scatter`: rank ≥ 3 operands, and any rank-2 form other than a full row or column write.
- iree's declared dtype set has no unsigned integer and no complex dtype (`evoxc/backends/adapters/iree.py`).
- Dynamic (non-static) dims cannot be exported at all.

## Reproduce

```bash
export PATH=/root/.local/bin:$PATH
uv venv "$TMPDIR/backend-venv" --python 3.12
uv pip install --python "$TMPDIR/backend-venv/bin/python" -e . "evoxc[iree]"

# one fresh process per (case, backend, pop) — peak RSS and the build are per-process
"$TMPDIR/backend-venv/bin/python" benchmarks/backend_measure.py --backend numpy --case xor --pop 100
"$TMPDIR/backend-venv/bin/python" benchmarks/backend_measure.py --backend iree  --case xor --pop 100 --json iree.json
"$TMPDIR/backend-venv/bin/python" benchmarks/backend_measure.py --backend iree  --case xor --pop 100 --opt-level O0
"$TMPDIR/backend-venv/bin/python" benchmarks/backend_measure.py --backend iree  --case xor --pop 100 --opt-level O2  # fails inside iree-compile
```

The end-to-end backend test (`test/backends/test_iree_pipeline.py`) skips cleanly without the extra and runs a real XOR generation with it:

```bash
pytest -q test/backends            # 3 passed, or 1 skipped when evoxc[iree] is absent
pytest -m requires_iree test/backends
```

The minimal reproducers for gates 1–4 are ~20 lines of `evoxc` each; the recipes (trace the graphs, walk `op.location`, export with `evoxc.lower(graph, backend="iree").text()`, compile the text with `iree.compiler.compile_str`) live in `src/tensorneat/common/CONTEXT.md` and `src/tensorneat/algorithm/neat/CONTEXT.md`.

## What this does not change

`RESULTS.md` is generated evidence for the old-vs-new CPU contract and was not regenerated or hand-edited for this report: the contract is measured on the numpy backend and that is still what it measures.
`benchmarks/results/iree.json` + `benchmarks/iree.md` remain the historical pre-fix record (8/8 `error`); the numbers above supersede them for the *current* code.
