# HyperNEAT Substrates

## Intent
A substrate is HyperNEAT **geometry**, not computation: it turns a neuron layout (one coordinate per neuron, in `cd` dimensions) into the fixed phenotype topology the CPPN is queried over — one CPPN query per phenotype connection.
Everything here is plain host-side Python/NumPy executed once at construction time; a substrate never holds a `State`, never runs inside an `evoxc` trace, and owns no tensors at trace time.
`../hyperneat.py` reads these arrays while tracing and freezes them as **graph constants**: `query_coors` becomes the CPPN's input batch, `nodes` and `conn_index` are packed into gene-payload rows by the genome's payload builder, and the CPPN output row `q` becomes the `weight` attribute of connection `q`.
Only the CPPN evolves; the topology here is immutable for the whole run.
Two usable implementations: `FullSubstrate` (recurrent full mesh) and `MLPSubstrate` (feedforward layered grid).

## API Surface — `BaseSubstrate` protocol (binding; the sibling `../hyperneat.py` codes against exactly this)
`BaseSubstrate` is a plain class — no `StatefulBaseClass`, no `setup`, no `make_nodes`/`make_conns`, no arrays as constructor arguments.
`connection_type: str` class attribute — `"feedforward"` or `"recurrent"`, the executor requirement of the phenotype (`None` on the abstract base).
`num_inputs -> int` property — input neuron count **including** the trailing bias neuron (`HyperNEAT` appends the bias `1.0` itself and reports `num_inputs - 1` upward).
`num_outputs -> int` property — output neuron count, also the phenotype's output width.
`nodes_cnt -> int` property — neuron count, equal to the hyper genome's `max_nodes`.
`conns_cnt -> int` property — connection count, equal to the hyper genome's `max_conns`.
`query_coors -> np.ndarray` property — `(conns_cnt, 2*cd)` `float32`; row `q` is `[coor_of_source, coor_of_target]` of connection `q`, so `query_coors.shape[1]` is the CPPN's required `num_inputs` (`../hyperneat.py::_validate` raises unless it equals `neat.num_inputs`).
`nodes -> np.ndarray` property — `(nodes_cnt,)` `int32` node keys; a dense permutation of `0 .. nodes_cnt-1` in the order input block, output block, hidden block(s), so genome index `k` is node key `k`.
`conn_index -> np.ndarray` property — `(conns_cnt, 2)` `int32` `(in_key, out_key)` rows aligned row-for-row with `query_coors`.
`validate() -> None` — abstract on the base; `DefaultSubstrate` implements it as the public re-check of the storage invariants, so constructors run it and user subclasses may re-run it after replacing topology fields.
`__repr__() -> str` — implemented once on the base: one-line host-side summary (class name, `connection_type`, the four counts); never dumps the arrays.
`__repr__` reads the counts through the properties, so it works for every concrete substrate and raises `NotImplementedError` only on a bare `BaseSubstrate`, which has no counts to report.
**No weight column exists anywhere in this node** — weights arrive later from the CPPN and are packed by the genome's gene payload builder, so this layer stays blind to gene layout.
`weight_threshold` and `max_weight` are `HyperNEAT` constructor knobs applied to the raw CPPN output in `../hyperneat.py::_weights_from_query` (a strict dead zone, then an unclipped `max_weight / (1 - weight_threshold)` scale); no substrate option reads them, and `activate_time` is likewise an executor knob, not a geometry knob.
`__init__.py` re-exports `BaseSubstrate`, `DefaultSubstrate`, `FullSubstrate`, `MLPSubstrate` plus the shared geometry helpers `as_coor_array`, `cartesian_product`, `check_same_coordinate_dim`, `layer_coors`, `node_key_blocks`.

## API Surface — shared geometry helpers (`base.py`)
`as_coor_array(coors, name="coors") -> np.ndarray` — copy and coerce one coordinate set to `(n, cd)` `float32` with `n >= 1`, `cd >= 1`; a 1-D input is rejected rather than promoted, so a single-neuron layout is written `((x, y),)`; non-finite coordinates and non-rectangular sequences are rejected too.
`check_same_coordinate_dim(*named_coors) -> int` — take `(name, array)` pairs and return the shared `cd`, raising if they disagree (or if no pair is given, or a pair is not 2-D with `cd >= 1`).
`node_key_blocks(num_inputs, num_outputs, hidden_sizes=()) -> (input_keys, output_keys, hidden_keys)` — emit the contiguous `int32` key blocks in TensorNEAT order, hidden keys concatenated layer by layer so hidden layer `i` occupies `hidden_keys[sum(hidden_sizes[:i]) : ...]`; a hidden size of `0` is legal and simply contributes no key.
It validates `num_inputs`/`num_outputs` as positive non-`bool` ints and every hidden size as a non-negative non-`bool` int, so `0` is the only allowed zero layer and a negative size raises.
`cartesian_product(keys1, keys2, coors1, coors2) -> (coors, key_pairs)` — fully connected bipartite block with the **source axis varying slowest**: row `i*len(keys2)+j` is source `keys1[i]` paired with target `keys2[j]`; `coors` is `(n1*n2, 2*cd)` `float32`, `key_pairs` is `(n1*n2, 2)` `int32`; an empty side yields `(0, 2*cd)` / `(0, 2)` with dtypes intact. It validates key/coordinate shapes and the shared `cd` before broadcasting, because a ragged block would silently misalign `query_coors`.
`layer_coors(neuron_cnt, x_min, x_max, y_coor) -> np.ndarray` — `(neuron_cnt, 2)` `float32` `(x, y)` pairs spanning `x_min..x_max` inclusively; a single neuron is centred at `((x_min+x_max)/2, y_coor)`.
`base.py` also owns the three private host-side validators every module imports: `_positive_int(value, name)` (rejects `bool`/float/`< 1`), `_as_float(value, name)` (rejects `bool`/non-numeric/non-finite) and `_to_int32(values, name)` (casts key arrays to a fresh `int32` copy, refusing fractional or non-numeric input).

## API Surface — concrete substrates
`DefaultSubstrate(num_inputs, num_outputs, query_coors, nodes, conn_index)` — `connection_type = "recurrent"`; the plain, user-subclassable container that stores already-computed topology.
`DefaultSubstrate` stores into private `_query_coors`/`_nodes`/`_conn_index`/`_num_inputs`/`_num_outputs` and exposes the inherited read-only properties; a base-class property without a setter would reject `self.nodes = ...` anyway.
`DefaultSubstrate` accepts `nodes` as `(nodes_cnt,)` or as the legacy `(nodes_cnt, 1)` column and flattens it; it copies and dtype-normalises every array once, because HyperNEAT later freezes them as constants.
`FullSubstrate(input_coors=((-1,-1),(0,-1),(1,-1)), hidden_coors=((-1,0),(0,0),(1,0)), output_coors=((0,1),))` — `connection_type = "recurrent"`; fully connected input→hidden, hidden→hidden (self-loops and both directions included) and hidden→output, with no direct input→output edges.
`MLPSubstrate(layers, coor_range=(-1, 1, -1, 1))` — `connection_type = "feedforward"`; one grid column per layer, adjacent layers fully connected, `coor_range = (x_min, x_max, y_min, y_max)`.
`FullSubstrate.analysis_substrate(input_coors, hidden_coors, output_coors)` and `MLPSubstrate.analysis_substrate(layers, coor_range)` — pure functions returning `(query_coors, nodes, conn_index)`; stateless, so tests and users can diff geometry without instantiating a substrate.
Both concrete constructors normalise their arguments into attributes first (`input_coors`/`hidden_coors`/`output_coors` as `float32`; `layers` as a `tuple[int, ...]` and `coor_range` as a four-`float` tuple), call their own `analysis_substrate`, and then hand `(num_inputs, num_outputs, query_coors, nodes, conn_index)` up to `DefaultSubstrate.__init__`, which normalises, stores and validates once.
`FullSubstrate.num_inputs`/`num_outputs` re-derive from the coordinate arrays and `MLPSubstrate`'s from `layers`; both agree with the stored `_num_inputs`/`_num_outputs` by construction.

## Geometry rules (ordering invariants — a silent reorder shows up only as bad fitness)
Node keys: input block `0 .. si-1` (bias neuron last), output block `si .. si+so-1`, hidden block(s) `si+so .. si+so+sh-1`; `nodes` is always a dense permutation of `0 .. nodes_cnt-1`.
`query_coors` row `q` and `conn_index` row `q` describe the **same** connection, and the CPPN output on row `q` is that connection's weight — this alignment is the single most important invariant of the node.
`FullSubstrate` row blocks, in this exact order: input→hidden (`si*sh` rows), hidden→hidden (`sh*sh` rows), hidden→output (`sh*so` rows); inside a block the source varies slowest.
`MLPSubstrate` row order: layer pairs ascending (`layer_idx = 0 .. layer_cnt-2`), then source neuron ascending, then target neuron ascending; there are no self-loops and no backward edges, hence `"feedforward"`.
`MLPSubstrate` coordinates: layer `0` at `y_min`, layer `-1` at `y_max`, hidden layer `i` at `y_min + i*(y_max-y_min)/(layer_cnt-1)`, every layer spread evenly over `[x_min, x_max]`.
`MLPSubstrate` coordinates are always 2-D, so `query_coors.shape[1] == 4`; `FullSubstrate` supports any `cd >= 1` because it inherits the coordinate width from the caller's arrays.
The last input coordinate is the bias neuron: `FullSubstrate`'s default three inputs are two real inputs plus bias, and `MLPSubstrate`'s `layers[0]` counts the bias neuron too (the XOR example passes `layers=[4, ...]` for 3 real inputs).
Pruning is expressed as zero CPPN-derived weights, never as missing genes: `max_nodes`/`max_conns` equal `nodes_cnt`/`conns_cnt`, so both arrays are dense and carry no NaN padding.

## Validation matrix (constructor-time, host-side, `TensorNEATError` never bare `assert`)
Every constructor raises `tensorneat.common.errors.TensorNEATError` — a configuration mistake must not surface as a NumPy/JAX shape error inside a later trace.
`BaseSubstrate`/`DefaultSubstrate`: `num_inputs >= 1` and `num_outputs >= 1` as integers (`bool` and floats rejected); `num_inputs + num_outputs <= nodes_cnt`; `query_coors` 2-D with a non-zero, **even** trailing width; `conns_cnt > 0`; `nodes` 1-D (or `(n, 1)`) with positive length, holding integer keys (a fractional float key is rejected, never truncated); `conn_index` shaped `(m, 2)` with `m >= 1`; `query_coors.shape[0] == conn_index.shape[0]`; node keys unique and forming a permutation of `0 .. nodes_cnt-1`; keys `0 .. num_inputs-1` are exactly the input block and `num_inputs .. num_inputs+num_outputs-1` the output block; every `conn_index` endpoint a known node key; `conn_index` rows unique (a duplicate row would mean two weights for one edge).
`FullSubstrate`: every coordinate set 2-D via `as_coor_array` (which also rejects empty sets, `cd == 0` and non-finite coordinates); all three share one `cd >= 1` via `check_same_coordinate_dim`; `input_coors` and `output_coors` non-empty; `hidden_coors` non-empty because the hidden layer is the only path to the outputs.
`MLPSubstrate`: `len(layers) >= 2`; every layer count a positive integer (bools, floats and strings rejected); `coor_range` holds four finite numbers with `x_min < x_max` and `y_min < y_max`.
`layer_coors` and `cartesian_product` validate too (positive `neuron_cnt`, 1-D key arrays, matching key/coordinate lengths, one shared `cd`), so the helpers raise the same error type whether called directly or through a substrate.
`MLPSubstrate` deliberately accepts the `len(layers) == 2` case (inputs wired straight to outputs) — that path is real, not degenerate.

## Constraints
No `jax`, `torch` or `evoxc` import in this node — plain `numpy` (+ `numpy.typing.ArrayLike`) only, so `import tensorneat` stays free of them (release gate).
Nothing here may construct a graph or touch `State`: substrates are created before `Pipeline.setup` and are pure data.
Arrays are materialised eagerly with NumPy — Python loops over the layer/index structure are correct here; this is host-side geometry, not a kernel.
All emitted arrays are `float32` (`query_coors`) and `int32` (`nodes`, `conn_index`), regardless of the input arrays' dtype.
Returned arrays are read-only in intent: HyperNEAT may already have frozen them as graph constants, so no caller may mutate them in place.
Construction must be cheap and side-effect-free, but it is allowed to be O(nodes × connections): the phenotype sizes are in the hundreds.
No `setup` method may appear — a subclass that needs state is a design error, not a missing hook.
Files stay near 500 lines and there is one reason to change per file: protocol + shared geometry + private validators (`base.py`, ~515 lines because the protocol docstrings are kept verbatim), storage + validation (`default.py`), recurrent mesh (`full.py`), feedforward grid (`mlp.py`).

## Divergences from the legacy JAX implementation (deliberate)
`StatefulBaseClass`, `setup`, `make_nodes(query_res)` and `make_conns(query_res)` are gone: arrays are exposed directly and the weight packing moved into the genome's gene payload builder inside `../hyperneat.py`.
`conns` (`(m, 3)` `float32` with a zero weight column) became `conn_index` (`(m, 2)` `int32`): the weight column was a lie this layer cannot fill, and float columns forced the genome to bitcast instead of storing keys.
`nodes` is emitted 1-D (`(nodes_cnt,)` `int32`) instead of the legacy `(nodes_cnt, 1)` `float64` column; the column is still accepted on input for legacy callers.
`full.analysis_substrate` takes `(input_coors, hidden_coors, output_coors)` — the legacy signature was `(input_coors, output_coors, hidden_coors)` while the constructor used the other order, a guaranteed transposition trap.
`cal_coors` became `layer_coors` and moved next to `cartesian_product` in `base.py`, so `full.py` and `mlp.py` share one implementation of the coordinate math.
Every legacy `assert` (and `DefaultSubstrate`'s total lack of shape checks) became an explicit `TensorNEATError`; a legacy configuration that silently produced a disconnected phenotype (`FullSubstrate` with empty `hidden_coors`, `total_conns == 0`) now raises.
`query_coors` is `float32` at the source instead of `float64` NumPy arrays relying on `jnp.array` to downcast later.
A multi-output CPPN still silently misaligns weights in the legacy `make_conns`; `../hyperneat.py` must therefore assert `neat.num_outputs == 1` — validation lives there because this node never sees the CPPN.

## Test Strategy
Unit tests belong to the sibling `test/` tree (read-only from here), mirroring this subtree at `test/unit/algorithm/hyperneat/substrate/`, on plain NumPy with no optional dependency.
Assertions worth having: dtype/shape of every array; `nodes` density and block order; `conns_cnt == query_coors.shape[0] == conn_index.shape[0]`; row-for-row alignment (`query_coors[q]` equals the concatenation of the two endpoint coordinates looked up from `conn_index[q]`); block/row order for both substrates; `MLPSubstrate`'s 2-layer case; every validation branch raising `TensorNEATError`.
Parity tests at `test/parity/algorithm/hyperneat/substrate/` may diff against the frozen `benchmarks/legacy_jax/tensorneat_legacy/algorithm/hyperneat/substrate/` geometry: `query_coors` must match the legacy arrays after `float32` cast, and `conn_index` must match legacy `conns[:, :2]` exactly — any difference is a bug, not a documented divergence.
No test in this node may build an evoxc graph or import `jax`.

## Consumer contract (`../hyperneat.py`, the parallel sibling)
It reads `substrate.query_coors` as the CPPN's `(conns_cnt, 2*cd)` input batch (one CPPN call per connection, weights mapped then packed), `substrate.nodes` and `substrate.conn_index` into gene payload rows whose leading columns are the keys, `substrate.nodes_cnt`/`conns_cnt` as the hyper genome's `max_nodes`/`max_conns`, `substrate.num_inputs`/`num_outputs` as the phenotype's fan-in/out (bias included, so report `num_inputs - 1` upward), and `substrate.connection_type` to pick the recurrent or feedforward executor.
It must assert `query_coors.shape[1] == neat.num_inputs` (CPPN input width) and `neat.num_outputs == 1` (scalar weight per connection).
It must never mutate the returned arrays and must not call anything in this node from inside a trace.

## Routing Table
This directory has no subdirectories — all four modules live directly here.
| Path | Responsibility |
| --- | --- |
| `./base.py` | `BaseSubstrate` protocol (abstract array/count properties + `validate`, concrete `__repr__`), the shared host geometry helpers (`as_coor_array`, `check_same_coordinate_dim`, `node_key_blocks`, `cartesian_product`, `layer_coors`) and the private validators `_positive_int`/`_as_float`/`_to_int32`. |
| `./default.py` | `DefaultSubstrate`: the validated container of a pre-computed topology — the base class users subclass for custom substrates. |
| `./full.py` | `FullSubstrate` + `analysis_substrate`: recurrent full mesh input→hidden→hidden→output. |
| `./mlp.py` | `MLPSubstrate` + `analysis_substrate`: feedforward layered grid, adjacent layers fully connected. |
| `../hyperneat.py` | Consumer: runs the CPPN over `query_coors`, maps its output to weights, and packs `nodes`/`conn_index` into gene payloads (sibling — read-only, escalate writes to the parent). |
