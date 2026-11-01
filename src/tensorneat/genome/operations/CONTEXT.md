# operations — batched genetic operators

## Intent
Provide mutation, crossover, and compatibility distance as batch-first kernels over the gene payload, plus the structural invariants that keep the execution schedule's static shapes valid.
Operators never allocate RNG keys or node/connection keys themselves: keys are drawn or allocated by the caller and passed in, which keeps runs reproducible for a `(seed, config, shape)` triple.

## API Surface
`BaseMutation.__call__(state, genome, key, node_keys, node_attrs, conn_keys, conn_attrs, new_node_keys (P,), new_conn_keys (P,3)) -> GenomeTensors`
- `mutate_structure(...)` applies at most one structural change per individual, gated by `node_add`/`node_delete`/`conn_add`/`conn_delete` probabilities evaluated elementwise from one `evoxc.random.uniform(key, (4, P))` draw.
- `mutate_values(...)` mutates every existing gene's attributes (one batched gene draw per array) and leaves padded rows untouched.
`DefaultMutation(conn_add=0.2, conn_delete=0.2, node_add=0.1, node_delete=0.1)` — NEAT-python style; add-conn rejects duplicates and, for `network_type == "feedforward"`, cycle-closing edges; both add paths refuse to exceed `max_conns`, `max_nodes`, or the genome's `max_in_degree`.
`RecurrentMutation(p_recur=0.1, max_conn_tries=20, **kwargs)` — biases add-conn toward cycle-forming edges by trying up to `max_conn_tries` candidate `(src, dst)` pairs; retains the `max_in_degree` cap.
`BaseCrossover.__call__(state, genome, key, genome1, genome2) -> GenomeTensors` — `genome1` is the fitter parent; each gene of `genome1` is kept or replaced by its key-identified homologue from `genome2`.
`DefaultCrossover` — alignment by exact key-column match (hash + sort window, never an `O(M²)` scan); float attributes cross per attribute via the gene's `crossover`.
`BaseDistance.__call__(state, genome, genome1, genome2) -> (P,)` — compatibility distance; `DefaultDistance(compatibility_disjoint=1.0, compatibility_weight=0.4)` normalizes by the larger gene count.

## Constraints
- All operators are trace-time kernels over `(P, ...)` payloads: no Python loop over individuals, no per-individual key splits, only elementwise/`gather`/`sort`/`argsort`/`reduce_*` ops.
- `argsort` has no AD rule, but operators are never differentiated; they must not, however, appear inside the `grad` graph.
- Structural invariants are enforced here, not silently repaired: `max_nodes`, `max_conns` and `max_in_degree` are hard caps, and an add that would break one is a no-op for that individual.
- Add-node is the only operation that changes topology while preserving the network output: it deletes the original connection, inserts the new node with the gene's identity attrs, adds an identity connection from the old source and a copy-of-original connection into the old destination.
- New rows go to the first free row (`first_free_row`) in the same layout for keys and attrs.
- Errors: invalid configuration raises `../common/errors.TensorNEATError`; no silent fallbacks.
- Files ≤ ~1000 lines; mutation's structural helpers are parameterized once and shared by `DefaultMutation`/`RecurrentMutation`.

## Routing Table
| Path | Responsibility |
|---|---|
| `./mutation/` | Structural/value mutation, `DefaultMutation`, `RecurrentMutation`, add/delete helpers usage. |
| `./crossover/` | `BaseCrossover`, `DefaultCrossover`, key alignment. |
| `./distance/` | `BaseDistance`, `DefaultDistance`, compatibility distance. |
