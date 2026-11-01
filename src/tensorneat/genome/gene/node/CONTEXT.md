# Intent
Node genes: one row of the `nodes` array (`shape (max_nodes, node_gene.length)`, float32, NaN-padded) describes one neuron keyed by column 0.
A node gene owns aggregation of incoming connection values, bias/response, and optional activation; `DefaultGenome`/`RecurrentGenome` default to `DefaultNode()`.

# API Surface
`BaseNode` (`base.py`) — `fixed_attrs = ["index"]`; `forward(state, attrs, inputs, is_output_node=False, valid_mask=None)`; host-side `repr`, `to_dict(state, node)` → `{"idx"}`, `sympy_func(state, node_dict, inputs, is_output_node=False)` → `(expr, {symbol: value})`.
`DefaultNode` (`default.py`) — `custom_attrs = ["bias", "response", "aggregation", "activation"]`, row length 5, NEAT-python semantics; ctor kwargs bias/response `{init_mean, init_std, mutate_power, mutate_rate, replace_rate, lower/upper_bound}`, `aggregation_*`, `activation_*` (options may be a single callable or a sequence; default `AGG.sum` / `ACT.sigmoid`; `*_default` becomes the index `options.index(default)`).
- `forward`: `z = apply_aggregation(agg, inputs, aggregation_options, valid_mask)`; `z = bias + response * z`; then `jax.lax.cond(is_output_node, z, apply_activation(act, z, activation_options))`, so output nodes are linear w.r.t. the aggregation result.
- `valid_mask=None` defaults to `~jnp.isnan(inputs)` (i.e. only real incoming connections count).
- `distance` = `|Δbias| + |Δresponse| + (agg1 != agg2) + (act1 != act2)`; `mutate`/`new_random_attrs` return `jnp.array([bias, res, agg, act])`.
- `sympy_func` builds `n_{idx}_b`, `n_{idx}_r` symbols, `AGG.obtain_sympy(nd["agg"])(inputs)`, `bias + res*z`, and applies `ACT.obtain_sympy(nd["act"])` unless `is_output_node`.
`BiasNode` (`bias.py`) — same as `DefaultNode` minus `response` (`custom_attrs = ["bias", "aggregation", "activation"]`, length 4); `forward` uses `z = bias + z`; `new_identity_attrs` returns `[0, aggregation_default, -1]`, i.e. it is the producer of the `activation == -1` (= `ACT.identity`) convention.
`OriginNode` (`origin.py`) — `DefaultNode` with `crossover` overridden to a single scalar coin flip (all custom attrs inherited from one parent, per the original NEAT paper), length 5.
References to "origin" mean original-NEAT-paper behavior (https://github.com/EMI-Group/tensorneat/issues/11), not un-activated/raw values.

# Constraints
`inputs` is the vector of `conn_gene.forward` results over *all* possible source nodes (length `max_nodes`) and `valid_mask` is the boolean mask of existing, already-computed sources; aggregation must ignore masked entries.
`is_output_node` is computed by the genome as `jnp.isin(nodes[i, 0], output_idx)` and must stay a traced value, so activation is switched with `lax.cond`, never a python `if`.
`act == -1` means identity in `apply_activation`, `repr` and `to_dict`; any new activation option should be an index into `activation_options`.
Aggregation indices index into `aggregation_options`, which defaults to the single-element list `[AGG.sum]` when a bare callable is passed.
`repr`/`to_dict`/`sympy_func` are host-side and must not be called on NaN rows.

# Known Issues
`DefaultNode.new_identity_attrs` returns `self.activation_default` while its comment claims `activation=-1`; only `BiasNode` emits `-1`.
`response_lower_bound` is stored as the misspelled `self.reponse_lower_bound` in both `DefaultNode.input`/`BiasNode` ctor and then used for clipping.

# Routing Table
No subdirectories.
