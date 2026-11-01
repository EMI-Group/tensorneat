# HyperNEAT

## Intent
HyperNEAT (hypercube-based NEAT) in JAX: a CPPN — an ordinary NEAT-evolved network from `tensorneat.algorithm.neat` — is queried once per substrate connection at the coordinates of its two endpoints, and its scalar output becomes that connection's weight.
The resulting substrate network (fixed topology, coordinates in N dimensions) is the phenotype that gets evaluated.
Two variants: `HyperNEAT` (recurrent substrate network, executed by `RecurrentGenome`) and `HyperNEATFeedForward` (feedforward substrate, executed by `DefaultGenome`).
Only the CPPN is evolved; the substrate topology and node genes are fixed at construction and the connection weights are re-derived from the CPPN on every `transform`.

## API Surface
`hyperneat.py` defines `HyperNEAT(BaseAlgorithm)`, `HyperNEATNode(BaseNode)`, `HyperNEATConn(BaseConn)`.
`HyperNEAT.__init__(self, substrate: BaseSubstrate, neat: NEAT, weight_threshold: float = 0.3, max_weight: float = 5.0, aggregation: Callable = AGG.sum, activation: Callable = ACT.sigmoid, activate_time: int = 10, output_transform: Callable = ACT.sigmoid)`.
`HyperNEAT.__init__` asserts `substrate.query_coors.shape[1] == neat.num_inputs` and builds `self.hyper_genome = RecurrentGenome(num_inputs=substrate.num_inputs, num_outputs=substrate.num_outputs, max_nodes=substrate.nodes_cnt, max_conns=substrate.conns_cnt, node_gene=HyperNEATNode(aggregation, activation), conn_gene=HyperNEATConn(), activate_time=activate_time, output_transform=output_transform)`.
`setup(state)` chains `neat.setup` → `substrate.setup` → `hyper_genome.setup` on the threaded `State`.
`ask(state)` / `tell(state, fitness)` / `show_details(state, fitness)` delegate straight to the wrapped `neat`.
`transform(state, individual)` returns `(nodes, conns, u_conns)` (the `RecurrentGenome` layout): it runs the CPPN as `vmap(self.neat.forward, in_axes=(None, None, 0))(state, transformed, substrate.query_coors)`, maps the CPPN output to a weight, then returns `hyper_genome.transform(state, substrate.make_nodes(query_res), substrate.make_conns(query_res))`.
Weight mapping in `transform`: `|w| < weight_threshold` → 0 (mutes/prunes near-zero connections), then shift by ∓`weight_threshold` for continuity, then scale by `max_weight / (1 - weight_threshold)`, giving weights in `[-max_weight, max_weight]`.
`forward(state, transformed, inputs)` appends a trailing bias `1.0` (`jnp.concatenate([inputs, jnp.array([1])])`) and delegates to `self.hyper_genome.forward`; the substrate's last input coordinate is the bias.
Property `num_inputs` = `substrate.num_inputs - 1` (bias removed), property `num_outputs` = `substrate.num_outputs`; `Pipeline` asserts `algorithm.num_inputs == problem.input_shape[-1]`.
`HyperNEATNode.__init__(aggregation=AGG.sum, activation=ACT.sigmoid)` holds fixed (non-evolved) aggregation/activation; `forward` applies `lax.cond(is_output_node, aggregation, activation∘aggregation)` so output nodes get no activation.
`HyperNEATConn` declares `custom_attrs = ["weight"]` and `forward` is `inputs * attrs[0]`, so substrate connection rows are `[in_idx, out_idx, weight]` with the weight overwritten every `transform`.
`hyperneat_feedforward.py` defines `HyperNEATFeedForward(HyperNEAT)` with `__init__(self, substrate, neat, weight_threshold=0.3, max_weight=5.0, aggregation=AGG.sum, activation=ACT.sigmoid, output_transform=ACT.sigmoid)` — it fully re-implements the parent constructor with `DefaultGenome` instead of `RecurrentGenome` (no `activate_time`) and asserts `substrate.connection_type == "feedforward"`; every other method is inherited.
Note the `transform` return arity differs by subclass: `HyperNEAT` → 3-tuple `(nodes, conns, u_conns)`, `HyperNEATFeedForward` → 4-tuple `(seqs, nodes, conns, u_conns)`.

## Constraints
`State` (`tensorneat.common.State`) is immutable and a registered pytree: `setup`/`tell` return a new `State` via `register`/`update`; never mutate it or any array in place.
Array updates are functional `.at[...].set(...)` only (e.g. `DefaultSubstrate.make_conns` replaces the weight column of the cached conn array).
All substrate shapes are static and materialized eagerly in `__init__` (NumPy arrays from Python loops converted with `jnp.array`), so substrate construction must happen outside any jit trace — the Python loops in `full.analysis_substrate` / `mlp.analysis_substrate` are not traceable.
`max_nodes` / `max_conns` of the hyper genome equal `substrate.nodes_cnt` / `substrate.conns_cnt`, so the substrate arrays are dense (no NaN padding) and pruning is expressed through zero weights rather than absent genes.
The hyper genome is never mutated, speciated or recombined — it is a fixed-topology executor that is rebuilt from CPPN outputs on each `transform`, so all evolution and all `randkey` consumption stay inside the wrapped `neat`.
No `@jax.jit` decorator exists anywhere in this subtree: `transform` and `forward` are traced by `Pipeline`, which compiles `jax.jit(self.step)` and `jax.jit(jax.vmap(self.algorithm.transform, in_axes=(None, 0)))` (`pipeline.py`), so these methods must stay pure and traceable.
`transform` assumes the CPPN has exactly one output: `DefaultSubstrate.make_conns` does `query_res.flatten()` onto the weight column, which silently misaligns for multi-output CPPNs.
Weight sign range is decided by the CPPN's own `output_transform` (set on `neat.genome`): `ACT.tanh` gives signed weights, whereas `ACT.sigmoid` gives non-negative weights only.
The `output_transform` argument of `HyperNEAT` is the substrate network's output activation, not the CPPN's.

## Known Issues
`HyperNEATNode.forward(self, state, attrs, inputs, is_output_node=False)` is out of sync with the gene protocol `BaseNode.forward(..., is_output_node=False, valid_mask=None)`: `DefaultGenome.forward` passes `valid_mask=` as a keyword while `RecurrentGenome.forward` passes it as a 5th positional argument, so both paths raise `TypeError` at trace time.
`HyperNEATNode.forward` calls `self.aggregation(inputs)` with a single argument, but every `AGG` function has signature `(z, mask)` (e.g. `sum_(z, mask)`), which raises `TypeError`; invalid-source masking (`apply_aggregation` / `valid_mask`) is not implemented here, unlike `DefaultNode.forward`.
`HyperNEATConn.forward` does not NaN-guard inputs or weights (unlike `DefaultConn.forward`), so invalid sources would propagate NaN if the node-side masking is ever bypassed.

## Test Coverage
No tests live in this subtree; `test/mlp_substrate.py` only prints `analysis_substrate` output.
Runnable examples: `examples/func_fit/xor_hyperneat.py` (`FullSubstrate`), `examples/func_fit/xor_hyperneat_feedforward.py` (`MLPSubstrate`), `examples/gymnax/cartpole_hyperneat.py`.

## Routing Table
| Path | Contents |
| --- | --- |
| `./substrate` | `BaseSubstrate` interface plus `DefaultSubstrate` / `FullSubstrate` / `MLPSubstrate`, including the coordinate → node/connection analysis and the `feedforward` vs `recurrent` `connection_type` contract. |
