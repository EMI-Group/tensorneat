# Intent
Tensor-valued (JAX) gene abstraction for TensorNEAT: a gene object is a stateless `StatefulBaseClass` descriptor that owns the layout and the operators for one fixed-width float32 row of a flattened gene array.
Gene objects hold no per-individual data; per-gene values live in `nodes` rows (width `node_gene.length`, shape `(max_nodes, length)`) and `conns` rows (width `conn_gene.length`, shape `(max_conns, length)`) created inside `genome/base.py`.
All gene arrays are float32, including index/marker columns, which are cast with `int()` at export time (`repr`, `to_dict`).

# API Surface
`BaseGene` (`base.py`), subclass of `common/stateful_class.py::StatefulBaseClass`:
- class attrs `fixed_attrs: list[str]` (identity/key columns) and `custom_attrs: list[str]` (evolvable columns); both are ordered python lists of names.
- property `length` = `len(fixed_attrs) + len(custom_attrs)` — the row width.
- `new_identity_attrs(state)` — custom attrs that make a gene behave as identity (used by "add node" mutation on the split connection; nan for `BaseGene`).
- `new_random_attrs(state, randkey)` — custom attrs sampled at initialization.
- `mutate(state, randkey, attrs)` — attrs are the custom-attr slice only.
- `crossover(state, randkey, attrs1, attrs2)` — default is per-element coin flip `jnp.where(jax.random.normal(randkey, attrs1.shape) > 0, attrs1, attrs2)` ("attrs exchange").
- `distance(state, attrs1, attrs2)` — scalar jnp for genome distance.
- `forward(state, attrs, inputs)` — one gene's numeric transform; returns a scalar.
- `repr(state, gene, precision=2)` and `hash(gene)` (full row → uint32 via `common/tools.py::hash_array`).
- `setup(state)`/`show_config()` come from `StatefulBaseClass`; genes are stateless so `setup()` just returns the passed `State`.
- `sympy_func` and `to_dict` are declared only on `BaseNode`/`BaseConn` (not on `BaseGene`); both host-side (python) and return sympy expressions + `{symbol: value}` or a plain dict.

# Index Layout Of A Gene Row
Columns `[0 : len(fixed_attrs)]` are the gene identity key, columns `[len(fixed_attrs) :]` are `custom_attrs` in declaration order.
`genome/utils.py::extract_gene_attrs(gene, row)` = `row[len(gene.fixed_attrs):]` and `set_gene_attrs` writes the same slice — the single source of truth for both directions.
Widths: `DefaultNode`/`OriginNode` 5 (`index,bias,response,aggregation,activation`), `BiasNode` 4 (`index,bias,aggregation,activation`), `DefaultConn` 3 (`input_index,output_index,weight`), `OriginConn` 4 (`input_index,output_index,historical_marker,weight`).
NaN padding: empty slots are the *entire* row set to NaN (`nodes.at[pos].set(jnp.nan)`); row existence is tested via the first column (`~jnp.isnan(row[0])`, `utils.valid_cnt`) and a new gene is inserted at the first all-NaN row (`fetch_first`).
Column order is load-bearing: `unflatten_conns` reads `conns[:,0]`/`conns[:,1]` as in/out node keys and `nodes[:,0]` as node key; `genome/base.py` writes `historical_marker` into column 2 when `"historical_marker" in conn_gene.fixed_attrs`; crossover/distance use `row[:len(fixed_attrs)]` as the matching key.

# Constraints
Everything except `repr`/`to_dict`/`sympy_func` runs inside jit/vmap/fori_loop: no python branching on traced values, no host transfers, attrs are jnp arrays of shape `(len(custom_attrs),)`.
Attribute order returned by `new_random_attrs`/`new_identity_attrs`/`new_zero_attrs`/`mutate` must match `custom_attrs` order, because genome/mutation code concatenates `fixed_attrs` with those arrays.
`AGG`/`ACT` indices are stored as floats in the row: aggregation is a plain index into `aggregation_options`, activation index `-1` means `ACT.identity` (see `common/functions/__init__.py::apply_activation`).
A conn gene may optionally expose `get_historical_marker(state, gene_array)`; `algorithm/neat/neat.py` and `genome/base.py` branch on `"historical_marker" in fixed_attrs` and otherwise number connections themselves.
Genes are shared objects across the whole population and must stay immutable after construction (mutation/crossover return new attrs, never in-place edits).

# JAX Usage
`forward` implementations use `jax.lax.cond`, `jax.lax.switch` (indirectly via `apply_activation`/`apply_aggregation`) and `jnp.isnan` masks; they are called through `vmap` from `DefaultGenome.forward` (per-node loop in `lax.fori_loop`) and `RecurrentGenome.forward` (per-timestep loop).
Aggregation receives a boolean `valid_mask` selecting which incoming connection values participate; `apply_aggregation` returns `0.0` when the mask is empty.
`hash_array` is jit-compatible (`lax.bitcast_convert_type` + `fori_loop` over `arr.size`), so gene hashing can run under jit.
Non-jittable/host-side: `repr`, `to_dict`, `sympy_func` (sympy symbols, `sp.lambdify`, `AGG.sympy_module(backend)`/`ACT.sympy_module(backend)` for `numerical_eval`), `__init__` option lists.

# Known Issues
`DefaultNode.new_identity_attrs` returns `self.activation_default` although its inline comment claims "activation=-1 means ACT.identity"; only `BiasNode.new_identity_attrs` actually emits `-1`.
Both `DefaultNode.__init__` and `BiasNode.__init__` store the `response_lower_bound` argument into a misspelled attribute `self.reponse_lower_bound`, which the clipping code then uses.
`DefaultConn.sympy_func` takes a `precision=None` parameter instead of the `is_output_node` argument used by node genes, so it is not signature-compatible with `BaseNode.sympy_func`.

# Routing Table
`./conn` — `BaseConn`, `DefaultConn`, `OriginConn`: connection genes (weight, optional historical marker).
`./node` — `BaseNode`, `DefaultNode`, `BiasNode`, `OriginNode`: node genes (bias/response/aggregation/activation).
