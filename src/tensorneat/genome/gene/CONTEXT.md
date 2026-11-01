# gene — attr-table driven gene families

## Intent
Define what a gene *is* (its key columns and its float attribute columns) and how it converts, mutates, crosses over, distances, and forwards, so that node and conn families are declarations over a shared mechanism instead of copy-paste.
`BaseGene` owns the table-driven plumbing; `node/` and `conn/` add only their structural hooks.

## API Surface
`BaseGene(StatefulBaseClass)`:
- class data: `fixed_attrs: list[str]` (int32 key columns), `attr_specs: tuple[GeneAttr, ...]` (float32 attribute columns, in row order).
- `key_width` / `attr_width` properties (`len(fixed_attrs)` / `len(attr_specs)`), `length` = `key_width + attr_width` for display only.
- `new_random_attrs(state, key, shape) -> (*shape, attr_width)` — one batched `evoxc.random` draw per attribute spec, clipped to the spec bounds.
- `new_identity_attrs(state) -> (attr_width,)` — attribute values that make the gene an identity transform (used by add-node splits); `new_zero_attrs(state)` for add-conn.
- `mutate(state, key, attrs) -> attrs` — batched over leading axes; per float spec `mutate_rate` noise / `replace_rate` re-init (legacy `mutate_float` distribution), per int spec `replace_rate` re-draw over the option indices (legacy `mutate_int`).
- `crossover(state, key, attrs1, attrs2) -> attrs` — per-attribute 50/50 pick; `OriginNode`/`OriginConn` override with a whole-gene pick.
- `distance(state, attrs1, attrs2) -> (...)` — sum over the attribute axis of `|Δ|` for floats and `!=` for ints.
- `hash(keys) -> (...,) uint32` — xor-mix over the int32 key columns (no bitcast available; hash is taken over keys, not float bits).
- Host-only: `repr`, `to_dict`, `sympy_func`.
`BaseNode.forward(state, attrs, messages, is_output_node, valid_mask) -> (P, N)` and `BaseConn.forward(state, attrs, source_values) -> (P, N, K)` are the only trace-time entry points.
`GeneAttr` is a small dataclass: `name`, `kind` (`"float"`/`"int"`), init mean/std or option index list, mutate power/rate, replace rate, optional lower/upper bound.

## Constraints
- Attribute order in `attr_specs` *is* the column order of `*_attrs`; never reorder without changing every consumer.
- `new_random_attrs` draws with the full batched shape in one call — one `evoxc.random.normal(key, shape)` per spec, never a per-row Python loop and never a per-row key split (evoxc keys are rank-0 only).
- Int attributes (aggregation/activation option indices) live in the float32 attr array and are compared with exact equality; `-1` means "identity activation" and is handled by `apply_activation`, not by an option-table index.
- No `jax`/`torch`; no eager NumPy on the compute path; NaN-free branches only, since every option is evaluated by a `select` chain.
- Files ≤ ~1000 lines; a new gene family adds declarations plus hooks, never a copy of the mutation/crossover code.

## Routing Table
| Path | Responsibility |
|---|---|
| `./base.py` | `GeneAttr` and `BaseGene`: table-driven attrs, batched mutate/crossover/distance/hash, host `repr`/`to_dict` hooks. |
| `./node/` | Node genes: hit-window aggregation, combine, activation selection. |
| `./conn/` | Conn genes: per-edge message from (attrs, source value). |
