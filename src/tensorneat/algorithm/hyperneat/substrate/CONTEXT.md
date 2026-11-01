# HyperNEAT Substrates

## Intent
A substrate fixes the phenotype's topology: it maps the coordinates of every connection endpoint to a row of `query_coors`, and turns the CPPN's per-row scalar output back into a dense node array plus a dense `[in_idx, out_idx, weight]` connection array for the `RecurrentGenome` / `DefaultGenome` executor.
Only the CPPN is evolved; everything here is static, materialized once at construction, and rebuilt from CPPN outputs on each `HyperNEAT.transform`.
Two usable implementations: `FullSubstrate` (recurrent, full hidden↔hidden mesh) and `MLPSubstrate` (feedforward layered).

## API Surface
`base.py`: `BaseSubstrate(StatefulBaseClass)` with class attribute `connection_type = None`, abstract `make_nodes(self, query_res)` and `make_conns(self, query_res)`, and abstract properties `query_coors`, `num_inputs`, `num_outputs`, `nodes_cnt`, `conns_cnt` (all raise `NotImplementedError`).
`default.py`: `DefaultSubstrate(BaseSubstrate)`, `connection_type = "recurrent"`, `__init__(self, num_inputs, num_outputs, coors, nodes, conns)` storing `self.coors/self.nodes/self.conns` as `jnp.array(...)` (inputs/outputs counts kept as `self.inputs`/`self.outputs`).
`DefaultSubstrate.make_nodes(query_res)` returns the cached `self.nodes` unchanged and ignores `query_res`.
`DefaultSubstrate.make_conns(query_res)` returns `self.conns.at[:, -1].set(query_res.flatten())` — a functional copy with the last column overwritten by the CPPN output.
`DefaultSubstrate` properties return `self.inputs`, `self.outputs`, `self.nodes.shape[0]`, `self.conns.shape[0]`, and `self.coors` for `query_coors`.
`full.py`: `FullSubstrate(DefaultSubstrate)`, `connection_type = "recurrent"`, `__init__(self, input_coors=((-1,-1),(0,-1),(1,-1)), hidden_coors=((-1,0),(0,0),(1,0)), output_coors=((0,1),))`, which calls `analysis_substrate(input_coors, output_coors, hidden_coors)` (note the argument order) and forwards `len(input_coors), len(output_coors)` to the parent.
`full.py`: `analysis_substrate(input_coors, output_coors, hidden_coors)` returns `(query_coors, nodes, conns)`: `query_coors` is `(total_conns, 2*cd)` float, `nodes` is `(si+so+sh, 1)` index, `conns` is `(total_conns, 3)` float32 with `conns[:, :2]` filled from the endpoint indices and column 2 zeroed.
`full.py`: `total_conns = si*sh + sh*sh + sh*so` over the blocks input→hidden, hidden→hidden, hidden→output (in that order).
`full.py`: `cartesian_product(keys1, keys2, coors1, coors2)` returns `(new_coors, correspond_keys)` with the `keys1`/`coors1` axis repeated `len(keys2)` times (outer-product order, first endpoint index varies slowest) and coordinates concatenated as `[coors1, coors2]` per row.
`mlp.py`: `MLPSubstrate(DefaultSubstrate)`, `connection_type = "feedforward"`, `__init__(self, layers: List[int], coor_range: Tuple = (-1, 1, -1, 1))` where `coor_range` is `(x_min, x_max, y_min, y_max)` and `layers` are per-layer neuron counts.
`mlp.py`: `MLPSubstrate.__init__` asserts `len(layers) >= 2`, every `layer > 0`, `coor_range[0] < coor_range[1]`, and `coor_range[2] < coor_range[3]`, then sets `num_inputs = layers[0]`, `num_outputs = layers[-1]`.
`mlp.py`: `analysis_substrate(layers, coor_range)` returns the same `(query_coors, nodes, conns)` triple, connecting only adjacent layers (`layer_idx` to `layer_idx + 1`), so there are no self- or backward connections.
`mlp.py`: `cal_coors(neuron_cnt, x_min, x_max, y_coor)` spreads a layer evenly over `x`, returning the single centered point `((x_min+x_max)/2, y_coor)` when `neuron_cnt == 1`.
`__init__.py` re-exports `BaseSubstrate`, `DefaultSubstrate`, `FullSubstrate`, `MLPSubstrate`; `tensorneat/algorithm/hyperneat/__init__.py` re-exports them again.

## Conventions
Node indices follow the TensorNEAT ordering input block → output block → hidden block (`0..si-1`, `si..si+so-1`, `si+so..`), which is why `nodes` is written as `concatenate(input_idx, output_idx, hidden_idx)` even though the layer list conceptually reads input→hidden→output.
Connection arrays always have exactly 3 columns `[in_idx, out_idx, weight]`, so the weight is the **last** column and `DefaultSubstrate.make_conns` can overwrite it blind.
`query_coors` row order must match `conns` row order 1:1 — row `i` of `query_coors` is the CPPN query whose output becomes `conns[i, -1]`.
`query_coors.shape[1] == 2 * coordinate_dim` must equal the CPPN's input count (`HyperNEAT.__init__` asserts this); the default `FullSubstrate` has 2-D coordinates, so 4 CPPN inputs.
`connection_type` is the contract between substrate and algorithm: `HyperNEAT` (recurrent genome) accepts `"recurrent"`, `HyperNEATFeedForward` asserts `"feedforward"` — a plain `DefaultSubstrate` is therefore unusable with `HyperNEATFeedForward` unless subclassed with `connection_type = "feedforward"`.
All substrate shapes are static and built eagerly with Python loops over NumPy arrays in `__init__`, so construction must happen outside any `jit`/`vmap` trace; nothing here is decorated with `jax.jit`.
Substrates are stateless: no `setup` override, so the inherited `StatefulBaseClass.setup` returns the `State` unchanged and the arrays live as plain attributes on the object.
`jnp.array` on float64/int64 NumPy input yields float32/int32 (JAX x64 is off), so indices and weights are 32-bit and `conns` from `full`/`mlp` are already `np.float32`.
Pruning is expressed as zero weights, not missing genes: `max_nodes`/`max_conns` of the hyper genome equal `nodes_cnt`/`conns_cnt`, so the arrays are dense and the CPPN's `weight_threshold` muting happens upstream in `HyperNEAT.transform`.
Sizes are small enough that connectivity is built by nested Python loops rather than vectorized NumPy.

## Known Issues / Gotchas
`make_conns` assumes exactly one CPPN output: `query_res.flatten()` lands on the weight column, so a multi-output CPPN misaligns weights silently instead of raising.
`DefaultSubstrate` does no shape validation: if `conns.shape[0] != coors.shape[0]`, the failure surfaces as a JAX shape error inside the jitted `transform` rather than at construction.
`DefaultSubstrate.make_nodes` discards `query_res` entirely, so CPPN output can never add or remove substrate nodes — node presence and ordering are fully caller-defined.
The `(query_coors, nodes, conns)` return order of `analysis_substrate` differs from the `make_nodes`/`make_conns` argument order of `transform`, and `full.analysis_substrate` takes parameters as `(input_coors, output_coors, hidden_coors)` while `FullSubstrate.__init__` takes `(input_coors, hidden_coors, output_coors)`.
The last input coordinate is the bias: `HyperNEAT.forward` appends a trailing `1.0` and `HyperNEAT.num_inputs = substrate.num_inputs - 1`, so `FullSubstrate`'s default 3 inputs are 2 real inputs + bias, and `MLPSubstrate`'s `layers[0]` likewise counts the bias neuron (examples pass `layers=[4, ...]` for the 3-input XOR problem).
`FullSubstrate` never creates direct input→output connections and treats the hidden layer as the only path, so an empty `hidden_coors` yields `total_conns == 0` and a completely disconnected substrate (no assertion guards this).
`FullSubstrate`'s hidden→hidden block is the full cartesian product, including self-loops (`i == j`) and both directions; this is what makes it `"recurrent"`.
`MLPSubstrate` is hard-wired to 2-D coordinates (`cal_coors` returns `(x, y)` pairs), whereas `full.analysis_substrate` supports arbitrary `cd` from the shape of the coordinate arrays.
`MLPSubstrate`'s `layers` place hidden layers at evenly spaced `y` between `y_min` and `y_max`, so `coor_range` must span at least the input and output rows or the layout degenerates.
`DefaultSubstrate`'s `connection_type` is `"recurrent"` even though it is the generic, caller-supplied-topology base class; the attribute describes the executor requirement, not the actual topology.
`default.py` has three unused imports: `vmap`, `numpy as np`, and `set_gene_attrs`.
`show_config()` inherited from `StatefulBaseClass` stringifies every attribute, so calling it on a substrate prints the full coordinate/node/connection arrays.

## Test Coverage
No dedicated tests: `test/mlp_substrate.py` merely prints `analysis_substrate([3, 4, 2], (-1, 1, -1, 1))` with no assertions.
Behavioral coverage comes from examples only: `examples/func_fit/xor_hyperneat.py` and `examples/gymnax/cartpole_hyperneat.py` (`FullSubstrate`), `examples/func_fit/xor_hyperneat_feedforward.py` (`MLPSubstrate`), plus the README HyperNEAT snippet.
Neither index-block ordering, endpoint ordering, nor the `query_coors`↔`conns` row alignment is asserted anywhere, so a silent reorder would only show up as bad fitness.

## Routing Table
This directory has no subdirectories — all four modules live directly here.
Parent node `./src/tensorneat/algorithm/hyperneat` holds the algorithms that consume these substrates (`hyperneat.py`, `hyperneat_feedforward.py`) and the CPPN-side weight mapping.
