# Genome (`src/tensorneat/genome`)

## Intent
Expose the variable-topology neural network of a NEAT individual as a pair of *fixed-shape, NaN-padded, fully jittable* JAX arrays, so an entire population can be `jit`/`vmap`ed.
`BaseGenome` owns configuration + algorithms-independent plumbing; `DefaultGenome` (`network_type="feedforward"`) and `RecurrentGenome` (`network_type="recurrent"`) are the two concrete network types.
There is no per-gene python object at runtime: a genome instance is just `nodes` + `conns` arrays plus the gene/operator objects.

## API Surface
`__init__.py` re-exports `gene.*`, `operations.*`, `BaseGenome`, `DefaultGenome`, `RecurrentGenome`.
`BaseGenome.__init__(num_inputs, num_outputs, max_nodes=..., max_conns=..., node_gene, conn_gene, mutation, crossover, distance, output_transform=None, input_transform=None, init_hidden_layers=())`.
`setup(state=State())` → `State` (chains `setup` of node_gene, conn_gene, mutation, crossover, distance; all stateless in the default implementation).
`transform(state, nodes, conns)` → `transformed` tuple, computed once per individual before evaluation.
`forward(state, transformed, inputs)` → `(num_outputs,)`; batch by `jax.vmap(self.forward, in_axes=(None, None, 0))`.
`grad(state, nodes, conns, inputs, loss_fn)` → `(loss, (grad_nodes, grad_conns))` via `jax.value_and_grad(argnums=(0,1))`, NaN-masked to 0 outside valid rows; `inputs.ndim==2` triggers a `vmap` over the batch.
`initialize(state, randkey)` → `(nodes, conns)` of the dense `num_inputs → init_hidden_layers → num_outputs` network.
`hash(nodes, conns)` → single `uint32` (per-row `common.tools.hash_array`, concatenated, hashed again).
`network_dict(state, nodes, conns, whether_re_cound_idx=True)` → python dict with `nodes`/`conns` dicts keyed by node key / `(in, out)`; DefaultGenome adds `topo_order`, `topo_layers`, `useful_nodes`.
`repr(state, nodes, conns, precision=2)`, `execute_mutation/execute_crossover/execute_distance(...)` (thin delegations to the operator objects), `get_input_idx()`/`get_output_idx()`.
`DefaultGenome.sympy_func(state, network, sympy_input_transform=None, sympy_output_transform=None, backend="jax")` → `(symbols, args_symbols, input_symbols, nodes_exprs, output_exprs, forward_func, topo_order, useful_nodes)`.
`DefaultGenome.visualize(network, rotate=0, reverse_node_order=False, ...)` writes an SVG with networkx + matplotlib (host only).

## Array Conventions
`nodes`: `(max_nodes, node_gene.length)` float32; col 0 = node key; unused rows are entirely NaN.
`conns`: `(max_conns, conn_gene.length)` float32; cols 0/1 = input/output node keys; col 2 = `historical_marker` only when `conn_gene.fixed_attrs` contains it; remaining cols = custom attrs.
`u_conns` from `utils.unflatten_conns(nodes, conns)`: `(max_nodes, max_nodes)` int32, `u_conns[src_row, dst_row]` = row index into `conns`, or `I_INF` when the connection is absent.
`cal_seqs` from `common.graph.topological_sort(nodes, conn_exist)`: `(max_nodes,)` int32, topological order of node **row** indices, remaining slots `I_INF`.
`vals`/`ini_vals`: `(max_nodes,)` float32, NaN means "not computed yet".
Row index equals node key for input rows `0..num_inputs-1` and output rows: `initialize` writes nodes in key order and input/output nodes are never deleted or moved.
Hidden nodes may live in any row (a freed row can be reused with a different key), so code compares keys via `nodes[i, 0]`, never row position.

## Forward Pass (feedforward, `default.py`)
`transform` returns `(cal_seqs, nodes, conns, u_conns)`; nothing is precomputed per node, so `forward` can be `vmap`ed over the population after a single vectorized `transform`.
`forward` builds `ini_vals` = NaNs with `ini_vals[input_idx] = input_transform(inputs)`, extracts `nodes_attrs (max_nodes, len(node_gene.custom_attrs))` and `conns_attrs (max_conns, len(conn_gene.custom_attrs))` with `vmap(utils.extract_gene_attrs)`.
`jax.lax.fori_loop(0, max_nodes, body_func, ini_vals)` runs a *fixed* `max_nodes` iterations; iteration `idx` handles node row `i = cal_seqs[idx]`, and `i == I_INF` is a no-op (`lax.cond`).
For a real, non-input node: `conn_indices = u_conns[:, i]` (all incoming conn row indices), `valid_mask = (conn_indices != I_INF) & ~isnan(vals)`, `hit_attrs = attach_with_inf(conns_attrs, conn_indices)`, `ins = vmap(conn_gene.forward, (None,0,0))(state, hit_attrs, vals)` (one weighted contribution per source row, NaN/absent sources contribute 0).
`node_gene.forward(state, nodes_attrs[i], ins, is_output_node=isin(nodes[i,0], output_idx), valid_mask=valid_mask)` aggregates over the source axis (masked), applies `bias + response*z` and the activation unless the node is an output; `lax.cond(any(valid_mask))` decides whether to write, so undecomposable nodes keep NaN and poison their descendants.
Output nodes are never activated by the node gene; activation of the final output is the user's `output_transform` (`forward` returns `vals[output_idx]`, or `output_transform(vals[output_idx])`).
`jnp.isin(i, self.input_idx)` short-circuits input nodes (they were seeded in `ini_vals`); `output_idx`/`input_idx` are keys used as row indices, valid by the row/key invariant above.

## Forward Pass (recurrent, `recurrent.py`)
`transform` returns `(nodes, conns, u_conns)` — no `cal_seqs`, because cycles are allowed.
`forward` runs `jax.lax.fori_loop(0, activate_time, ...)` (ctor arg, default 10) ticks of a synchronous (Jacobi) update: every existing node is recomputed from the previous tick's values, then `jnp.where(node_exists & any(valid_mask, axis=1), new_values, values)`.
`expand_conns_attrs = attach_with_inf(conns_attrs, u_conns)` gathers the weight of every `(src, dst)` pair, `conn_exists = (u_conns != I_INF).T` so masks are `(dst, src)`, and `node_ins.T` puts the destination axis first before `vmap(node_gene.forward)`.
Inputs are re-injected every tick (`values.at[input_idx].set(inputs)`), so the input node gene is bypassed exactly as in the feedforward path.
The loop is a fixed-depth relaxation, not convergence: `activate_time` (default 10) ticks exactly, and tick `t` can propagate information at most `t+1` edges from the inputs — a network deeper than `activate_time` leaves output rows NaN.
`vals` starts as all-NaN, so a node is written only once one of its sources is non-NaN; nodes with no path from an input stay NaN forever and poison descendants.
`valid_masks` are computed after re-injection each tick, so an edge from an input becomes usable in the same tick; output rows are recomputed like any other node, and `is_output_nodes` is recomputed inside the loop from `nodes[:, 0]`.
No edge is ever added into an input node (`choose_node_key(..., allow_input_keys=False)` for the target), so injected input values are never overwritten inside a tick.
`RecurrentGenome` does **not** apply `self.input_transform` (only `DefaultGenome.forward` does), and never builds `cal_seqs`.
`RecurrentGenome.sympy_func`/`visualize` raise `ValueError` (unsupported for cyclic graphs).

## Gene Variants and RecurrentMutation
`OriginNode` is `DefaultNode` with only `crossover` overridden: one scalar `jax.random.normal(randkey) > 0` picks the whole attribute vector (bias/response/agg/act together), instead of the per-attribute draw in `BaseGene.crossover`; attr layout, identity/random attrs, mutation powers (`bias/response` power 0.15, rate 0.2, replace 0.015) and `forward` are inherited unchanged.
`OriginConn` is `DefaultConn` plus `fixed_attrs = ["input_index", "output_index", "historical_marker"]` (row = `[in, out, marker, weight]`), the same whole-gene `crossover`, `get_historical_marker(gene) -> gene[2]`, and `repr`/`to_dict` that also print `historical_marker`; weight defaults/mutation are inherited.
`RecurrentMutation` subclasses `DefaultMutation` and inherits `__call__` (which asserts scalar `new_node_key` and `new_conn_keys.shape == (3,)`) and `mutate_values` unchanged; it overrides only `mutate_structure`.
Its scheduling draws all four probabilities from one key (`split(randkey, 5)` → `uniform(k_schedule, (4,))`) instead of splitting into four keys, and it gates each branch with `(p > 0) & (probs[i] < p)` rather than a python `if`; same order: node_add, node_delete, conn_add, conn_delete.
Its add-connection branch is retry-based: up to `max_conn_tries` (default 20) `(src, dst)` draws (src may be input or output, dst never input), first valid draw accepted, duplicate pairs rejected.
Cycle checking is not skipped but re-purposed: `p_recur` (default 0.1, validated `[0,1]`) forces a cycle — with probability `p_recur` an attempt is accepted only if `check_cycles` says it closes one, otherwise any non-duplicate pair is accepted (cycles allowed, as in `DefaultMutation`'s recurrent branch).
`conns_exist_matrix = unflatten_conns(...) != I_INF` is computed once outside the attempt `vmap`, and free space (`remain_conn_space >= 1`) is checked once up front, so retries do not re-check capacity.
All other branches (add-node, delete-node, delete-conn) are byte-for-byte the `DefaultMutation` logic, re-implemented inline as closures.

## Constraints
Everything on the evaluation path must be jit-safe: no python branching on traced values, no `device_get`, no python containers; use `lax.cond`/`lax.fori_loop`/`lax.switch`/`jnp.where`.
Pure-python (host-side, not jittable) helpers: `utils.re_cound_idx`, `utils.valid_cnt`, `repr`, `network_dict`, `sympy_func` (sympy + `lambdify`), `visualize` (networkx/matplotlib).
`transform` on a cyclic graph silently truncates `cal_seqs` (the `topological_sort` `while_loop` stops when no node has in-degree 0), so cycles belong to `RecurrentGenome` only.
`network_dict` on a cyclic graph calls `topological_sort_python`, which raises `ValueError("Graph has at least one cycle...")`.
`unflatten_conns` looks up conn endpoints by exact key match (`_batch_key_to_index`, argsort + searchsorted, O(C log N)); NaN endpoints map to `I_INF`, and absent pairs keep `I_INF` (int32 sentinel — NaN cannot be used in int32 arrays).
`attach_with_inf(arr, idx)` expands `idx` to `arr.ndim + idx.ndim - 1` dims and gathers `arr[idx]`, so gathered values keep a trailing custom-attr axis (length 1 for `DefaultConn`); aggregations therefore reduce over axis 0 (the source axis), not over the attr axis.
`BaseGenome.sympy_func()`/`visualize()` are declared with no arguments while the subclasses implement `(state, network, ...)` / `(network, ...)`; call the concrete class API.
`initialize` raises `ValueError` when `max_nodes`/`max_conns` cannot hold the initial dense network, and rejects `input_transform`/`output_transform` that fail on a zero vector.
`grad` NaN-masks the gradients (`jnp.where(isnan(nodes), 0, grad)`), so padded rows never accumulate NaN but valid rows connected to them can still legitimately be NaN-free only if the network is connected.

## Routing Table
| Path | Responsibility |
| --- | --- |
| `./gene/` | `BaseGene`/`BaseNode`/`BaseConn` + `DefaultNode`, `BiasNode`, `OriginNode`, `DefaultConn`, `OriginConn`: attr layout, `forward`, `sympy_func`, gene-level distance/crossover. |
| `./gene/node/` | Node gene variants and their activation/aggregation handling. |
| `./gene/conn/` | Connection gene variants (weight, optional `historical_marker`). |
| `./operations/` | `BaseMutation`/`DefaultMutation` (+ unexported `RecurrentMutation`), `BaseCrossover`/`DefaultCrossover`, `BaseDistance`/`DefaultDistance`. |
| `./operations/mutation/` | Structural + value mutations, add-node/add-conn, cycle checking. |
| `./operations/crossover/` | Historical-marker-based gene alignment. |
| `./operations/distance/` | Compatibility distance (excess/disjoint + weight difference). |

Out of node but required by this module: `../common/graph.py` (`topological_sort`, `topological_sort_python`, `find_useful_nodes`, `check_cycles`), `../common/tools.py` (`I_INF`, `attach_with_inf`, `fetch_first`, `fetch_random`, `hash_array`, `mutate_float`, `mutate_int`), `../common/functions/` (`ACT`, `AGG`, `apply_activation`, `apply_aggregation`), `../common/state.py`+`stateful_class.py` (`State`, `StatefulBaseClass`).
Consumers: `../algorithm/neat/neat.py` (`transform`/`forward` wrappers, `network_type`-dependent historical markers), `../algorithm/hyperneat/hyperneat.py` (uses `RecurrentGenome` as the CPPN), `../pipeline.py`, `../common/evox_adaptors/algorithm_adaptor.py`.
