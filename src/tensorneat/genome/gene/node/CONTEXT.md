# gene/node — node genes

## Intent
Turn one node's aggregated incoming-message window into that node's value, with the activation/aggregation choice stored in the gene row.
`DefaultNode` is the NEAT-python behaviour (`bias`, `response`, `aggregation`, `activation`); `BiasNode` drops `response`; `OriginNode` keeps `DefaultNode` attrs and only changes crossover to a whole-gene pick.

## API Surface
`BaseNode(BaseGene)`:
- `fixed_attrs = ["index"]` (one int32 key column, always the node key).
- `forward(state, attrs, messages, is_output_node, valid_mask) -> (P, max_nodes)`: `z = combine(attrs, aggregate(attrs, messages, valid_mask))`, then apply the activation only where `is_output_node` is False.
- `sympy_func(state, node_dict, inputs, is_output_node)` and `to_dict(state, key_row, attr_row) -> {"idx", "bias", "agg", "act", ...}` for host code generation.
Concrete hooks implemented by families: `combine(...)` (`bias + response * z` vs `bias + z`) and the attr spec tables.
`DefaultNode(..., aggregation_options=AGG.sum, activation_options=ACT.sigmoid, ...)` keeps the NEAT-python constructor knobs: bias/response init mean/std, mutate power/rate, replace rate, lower/upper bounds, plus `aggregation_default`, `activation_default`, `*_replace_rate`.
`BiasNode` has no `response` column and `bias + z` combine; its identity attrs use the default activation (**not** `-1`).
`OriginNode(BiasNode-compatible layout of DefaultNode)` overrides `crossover` only: one scalar draw selects the whole attribute vector.

## Constraints
- The aggregation reduces the **last axis** (the `K` in-edge window) after dead slots are replaced by that option's neutral; the gene must never see NaN (`messages` is already sanitized to 0 where the source value was NaN).
- Output nodes are never activated here; the final activation is `output_transform` at the genome level.
- Option indices must stay inside `[0, len(options))` or be exactly `-1` (identity); mutation re-draws from the configured option indices only.
- All forward code is elementwise over the population axis — no branching on traced values, no per-node Python loop.
- Files ≤ ~1000 lines.

## Routing Table
| Path | Responsibility |
|---|---|
| `./base.py` | `BaseNode`: `fixed_attrs`, `forward` contract, `repr`/`to_dict` scaffolding. |
| `./default.py` | `DefaultNode`: bias/response/aggregation/activation specs, `combine`, sympy. |
| `./bias.py` | `BiasNode`: `DefaultNode` minus `response`, declared over the same machinery. |
| `./origin.py` | `OriginNode`: `DefaultNode` attrs with whole-gene crossover. |
