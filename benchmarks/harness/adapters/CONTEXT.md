# harness/adapters — one adapter per implementation
## Intent
Hide the three real differences between the frozen JAX library and the evoxc rewrite — constructor arguments, compile model, sync/materialisation — behind the single `Adapter` contract documented in `../CONTEXT.md`.
`__init__.py` exposes `get_adapter(impl)` and must stay import-safe (no `jax`, no `tensorneat`, no `evoxc` at module scope): it imports `./old` or `./new` lazily, after the worker has fixed `sys.path`.
## API Surface
- `get_adapter(impl: str) -> Adapter` — `"old"` or `"new"`; unknown values raise `ValueError`.
- `old.py` — `OldAdapter`: imports `jax` and `tensorneat_legacy` at module scope (this module is only ever imported for the old side), `backend_label = "jax"`, build ignores the requested evoxc backend, `prepare` compiles `jax.jit(pipeline.step)` and consumes no generation.
- `new.py` — `NewAdapter`: adds `src/` to `sys.path`, imports `tensorneat`, asserts `jax`/`torch` are absent from `sys.modules`, forwards `backend=` into `Pipeline(...)` together with `is_detailed=False`, `prepare` calls `pipeline.step(state)` once and reports 1 consumed generation.
## Constraints
- Each adapter normalises its own result to a contiguous `float64` numpy array of shape `(P,)` before returning it — the caller never inspects JAX/evoxc types.
- Sync lives inside `step_fn`: the old side calls `jax.device_get` on the fitness array, the new side relies on evoxc's host materialisation; a future async backend must be synced here rather than in the worker.
- Failures propagate (no catching, no fallback): a build/compile error must reach the worker and be recorded as `status="error"`.
## Routing Table
| Path | Purpose |
|---|---|
| `./old.py` | Frozen JAX baseline adapter (`tensorneat_legacy`). |
| `./new.py` | evoxc rewrite adapter (`src/tensorneat`), includes the no-JAX release gate. |
| `../CONTEXT.md` | Adapter contract, fitness-extraction contract, JSON schema. |
