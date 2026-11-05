# Benchmarks — legacy JAX baseline

## Intent
Frozen copy of the **pre-rewrite** TensorNEAT implementation (JAX), kept only so the new evoxc implementation can be timed against it and diffed against it numerically.
This directory is not part of the shipped distribution (packaging finds packages under `src/` only) and must never be imported by `src/tensorneat`.

## API Surface
`import tensorneat_legacy` after putting `benchmarks/legacy_jax` on `sys.path` — the package is a rename of the original `tensorneat` (all internal imports rewritten), so every legacy module path works with the `_legacy` suffix:
`tensorneat_legacy.pipeline.Pipeline`, `tensorneat_legacy.algorithm.neat.NEAT`, `tensorneat_legacy.genome.DefaultGenome`, `tensorneat_legacy.problem.func_fit.XOR`, `tensorneat_legacy.common.{ACT,AGG,State}`.

## Configuration Defaults (the reference the rewrite is compared against)
### `algorithm.NEAT.__init__` (`algorithm/neat/neat.py:14-29`)
```python
def __init__(self, genome, pop_size, species_size=10, max_stagnation=15,
             species_elitism=2, spawn_number_change_rate=0.5, genome_elitism=2,
             survival_threshold=0.1, min_species_size=1, compatibility_threshold=2.0,
             species_fitness_func=jnp.max, species_number_calculate_by="rank",
             pop_batch_size=None)
```
- NEAT owns no mutation or gene rates; it only forwards the species knobs to `SpeciesController` (`neat.py:58-70`) — every rate lives on the genome's `DefaultMutation`/`DefaultNode`/`DefaultConn` defaults.
- `species_size` is the maximum number of species slots, not a members-per-species count: `species_arange = np.arange(species_size)` (`algorithm/neat/species.py:33`), the species state arrays have that length (`species.py:44-77`), `speciate` stops creating species at `i < species_size` and dumps still-unassigned genomes into the last species (`species.py:462-471`, `557-559`).
- `pop_size` is split among species by rank linear ranking: `rank_score = valid_species_num - arange`, `denominator = (N+1)*N/2`, `spawn = previous + (target - previous) * spawn_number_change_rate`, clamped by `min_species_size`, truncated to int, and the rounding error is added to the rank-0 species (`species.py:244-288`); species fitness is `jnp.max` over its members (`species.py:136-151`).
- `survival_threshold` is the fraction of each species eligible as parents: `survive_size = floor(survival_threshold * members_num)`, selection probability `(p_idx < survive_size) / survive_size` (`species.py:349-360`).
- `genome_elitism` individuals per species — the first `genome_elitism` slots of each species block — are the unmutated crossover result of that species' top members (`species.py:362-366`, `neat.py:184-186`).
- Stagnation drops a species after `> max_stagnation` generations without improvement, but the top `species_elitism` species never stagnate (`species.py:160-196`).
### `genome.DefaultGenome.__init__` (`genome/default.py:29-43`)
```python
def __init__(self, num_inputs, num_outputs, max_nodes=50, max_conns=100,
             node_gene=DefaultNode(), conn_gene=DefaultConn(), mutation=DefaultMutation(),
             crossover=DefaultCrossover(), distance=DefaultDistance(),
             output_transform=None, input_transform=None, init_hidden_layers=())
```
- `init_hidden_layers=()` means a fresh genome is the minimal feed-forward topology: no hidden nodes and a full bipartite connection between consecutive layers only (`genome/base.py:49-67`, `159-207`); XOR3d starts with nodes 0,1,2 (input) + 3 (output) and 3 connections, XOR with 2 connections.
- Initial population = `pop_size` identical minimal topologies whose attributes are drawn independently (`neat.py:79-82`, `genome/base.py:166-205`); topology only ever appears through mutation.
- `output_transform` defaults to `None` (raw summed output), and output nodes never apply their node-gene activation — only non-output (hidden) nodes do (`genome/gene/node/default.py:148-160`, `genome/default.py:127-130`).
### Gene / operator defaults
- `DefaultMutation`: `conn_add=0.2, conn_delete=0.2, node_add=0.1, node_delete=0.1` (`genome/operations/mutation/default.py:22-28`), applied as independent per-generation structural coin flips (`mutation/default.py:217-241`).
- `DefaultNode`: bias init N(0,1) clipped to ±5 with mutate rate 0.2 / power 0.15 / replace rate 0.015; response init 1.0 with std 0.0 and the same rates; aggregation options `[AGG.sum]`; **activation default is `ACT.sigmoid`** — `activation_options: ... = ACT.sigmoid` and `activation_default=None` resolves to `activation_options[0]` (`genome/gene/node/default.py:25-58`); activation replace rate 0.1 is a no-op while only one option is configured.
- `DefaultConn`: weight init N(0,1) clipped to ±5, mutate rate 0.2 / power 0.15 / replace rate 0.015, bounds ±5 (`genome/gene/conn/default.py:13-22`).
- `mutate_float`: with probability `mutate_rate` add N(0, `mutate_power`), with probability `replace_rate` replace by N(init_mean, init_std) (`common/tools.py:55-76`).
- `DefaultCrossover`: per-gene 50/50 uniform choice (`jax.random.normal > 0`) and disjoint genes always from the winner (`genome/gene/base.py:26-32`, `genome/operations/crossover/default.py:71-`).
- `DefaultDistance`: `compatibility_disjoint=1.0`, `compatibility_weight=0.4`, normalised by the larger gene count (`genome/operations/distance/default.py:9-15`, `64-68`).
### `problem.func_fit.XOR3d` (`problem/func_fit/xor3d.py:8-35`, `func_fit.py:12-42`)
- 8 rows `000..111` and 3-bit-parity targets `[0,1,1,0,1,0,0,1]`, both `(8,1)` float32; default `error_method="mse"` and fitness `= -mean((predict - targets)**2)`.
- Parity is orthogonal to every linear direction, so the best linear read-out is the constant 0.5 → mse 0.25 → fitness `-0.25`; a zero-weight genome under the sigmoid read-out sits exactly there, which is why `fitness_target=-0.05` is reachable only by a nonlinear network.
### Seed consumption (`pipeline.py`)
- `Pipeline.__init__` calls `np.random.seed(self.seed)` for the legacy host-side dataset draws (`pipeline.py:51`).
- `Pipeline.setup` registers `randkey = jax.random.PRNGKey(self.seed)` (`pipeline.py:84`); `NEAT.setup` splits it into `k1, randkey` and splits `k1` into `pop_size` initialisation keys (`neat.py:76-82`); `Pipeline.step` splits `randkey` into one evaluation key per individual per generation.
### `common.ACT` (`common/functions/__init__.py:9-58`)
- Registered activations: `scaled_sigmoid, sigmoid, scaled_tanh, tanh, sin, relu, lelu, identity, inv, log, exp, abs` (`clip` exists only as a sympy symbol).
- `ACT` has no global default; the default an unnamed genome config gets is `DefaultNode.activation_options[0]` = `ACT.sigmoid`.
- `apply_activation` treats index `-1` as identity and otherwise `lax.switch`es over the configured list (`common/functions/__init__.py:61-75`).
### What the benchmark case dict leaves at default
- `benchmarks/cases.py` sets only `pop_size`, `species_size=20`, `survival_threshold=0.01` for NEAT and `num_inputs`, `num_outputs=1`, `max_nodes=50`, `max_conns=100`, `output_transform=sigmoid` for the genome.
- Left at default and behaviour-relevant: `init_hidden_layers=()` (minimal topology), `activation_options=[sigmoid]`, weight/bias/response mutation rates and powers, `conn_add/node_add/conn_delete/node_delete` (0.2/0.1/0.2/0.1), `compatibility_threshold=2.0`, `max_stagnation=15`, `species_elitism=2`, `genome_elitism=2`, `min_species_size=1`, `spawn_number_change_rate=0.5`, `species_number_calculate_by="rank"`, `pop_batch_size=None`.
## Provenance
Provenance and the source commit are recorded in `PROVENANCE.md`; treat this tree as read-only.

## Constraints
- FROZEN: never edit, refactor, or format code here, and never "fix" its bugs — a moving baseline makes the benchmark meaningless.
- Requires `jax` (the `bench` extra); nothing else in the repo may depend on it.
- Runs are spawned in subprocesses by the harness so JAX threads/allocators cannot perturb the evoxc side of a measurement.
