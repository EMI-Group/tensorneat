# rl — reinforcement-learning problems on evoxc

## Intent
Score a genome by the return of a batched episode rollout, with the environment dynamics written as evoxc ops instead of a JAX simulator.
Three environments ship: the discrete `CartPole` and the continuous classic-control tasks `Pendulum` and `MountainCarContinuous`.
Each one doubles as the reference recipe for any environment a user wants to add: no registry, no simulator dependency, just a step kernel and one loop, one file per environment.

## API Surface
- `cartpole.py: CartPole(BaseProblem)` — `__init__(max_step=500, repeat_times=1, gravity=9.8, masscart=1.0, masspole=0.1, length=0.5, force_mag=10.0, dt=0.02, x_threshold=2.4, theta_threshold=12.0, reset_scale=0.05, fitness_target=None)`; every knob is validated on the host and raises `TensorNEATError`.
- `pendulum.py: Pendulum(BaseProblem)` — `__init__(max_step=200, repeat_times=1, gravity=10.0, mass=1.0, length=1.0, max_speed=8.0, max_torque=2.0, dt=0.05, fitness_target=None)`; every knob validated on the host (`require_positive_int`/`require_positive` from `_common`), `fitness_target > 0` raises because every Pendulum reward is `-cost <= 0`, and the attribute is stored as given.
- `mountaincar.py: MountainCarContinuous(BaseProblem)` — `__init__(max_step=999, repeat_times=1, min_position=-1.2, max_position=0.6, max_speed=0.07, goal_position=0.45, goal_velocity=0.0, power=0.0015, gravity=0.0025, fitness_target=None)`; unreachable geometry (`min_position >= max_position`, a `goal_position` outside the clipped track, `goal_velocity > max_speed`) raises as a configuration error, and so does `fitness_target > 100`.
- `_common.py` — the host-side helpers all three environments draw from: `to_numpy(value)`, the owner-prefixed validators `require_positive_int(owner, name, value)`, `require_positive(owner, name, value)` and `require_non_negative(owner, name, value)` (messages are `<Owner>: <name> must be ...`), `returns_of(total, pop_size, repeat_times)`, the `(B,) -> (P, R)` layout helper, and `prepared_query(act_fn, batch)`; the last two are **traced** (they emit ops), inlined into the caller's graph like `problem.base.finalize_fitness`.
  Each environment imports only what it uses: `pendulum.py` takes `require_positive_int`/`require_positive` (every knob is strictly positive), `mountaincar.py` adds `require_non_negative` for its `gravity`, and `cartpole.py` takes all three plus `to_numpy`/`returns_of`; all three take `to_numpy` and `returns_of`.
  No module-scope `enp` call and no `@evoxc.defn` lives there.
- `CartPole`: `input_shape = (4,)` (`x, x_dot, theta, theta_dot`), `output_shape = (2,)` (two logits; `out[1] > out[0]` pushes right).
- `Pendulum`: `input_shape = (2,)` (`theta, theta_dot`), `output_shape = (1,)` (one torque logit, squashed inside the problem).
- `MountainCarContinuous`: `input_shape = (2,)` (`position, velocity`), `output_shape = (1,)` (one action logit, `a = tanh(logit)`).
- Both continuous problems own the action squashing, so a genome used with them must leave `output_transform` unset (identity); a bounded transform would rescale the logit before `tanh` ever sees it.
- `fitness_target` is validation-only in all three: `None` skips the check, a value above the return ceiling raises.
- `evaluate(state, keys, act_fn, transformed) -> (P,)`: fitness is maximized, the mean over `repeat_times` of the episode return, `finalize_fitness`-ed last.
- `show(state, key, act_fn, transformed)`: host-side, one individual, builds its own graph over the single rank-0 key (`evoxc.build` + `evoxc.run`) and prints one line per repeat plus the mean return; returns `None`.
- `CartPole` trace-side helpers: `_rollout`, `_loop_cond`, `_loop_body`, `_constants`, `_advance`, `_death_mask`, `_terminated`, `_force_from_logits`, `_nan_rows`; validation and the host conversion are the shared `_common` names (`require_positive_int`/`require_positive`/`require_non_negative` with owner `"CartPole"`, `to_numpy`, `returns_of`).
- `Pendulum` trace-side helpers: `_rollout` (reset draw + loop → `(total, final_obs)`), `_reset` (the single `U(-1, 1)` draw, per-column scaled), `_loop_cond`, `_loop_body`, `_constants` (returns the `_PendulumConstants` NamedTuple), `_advance`, `_angle_normalize`.
- `MountainCarContinuous` trace-side helpers: `_rollout`, `_reset`, `_loop_cond`, `_loop_body`, `_constants` (`_MountainCarConstants`), `_action_from_logits`, `_advance`, `_goal_mask`, `_terminated` (rank-1 façade kept for host-side introspection and tests).

## Algebra

### Shared shape (all three)
`B = P * repeat_times` with `P = static_batch_size(transformed)`, **individual-major**: row `p * R + r` is repeat `r` of individual `p`, so `P = 4, R = 1` and `P = 2, R = 2` draw the same four initial states from one key and produce identical per-episode returns (verified for both continuous environments).
Initial states come from **one** `evoxc.random.uniform` from the single rank-0 key; the key is not threaded through the loop and `np.random` is never used.
The return accumulator rides in the carry and the loop emits no output: `evoxc.scan` would desugar to a `while` that carries a `(max_step, B)` stacked output and scatters one row per iteration, i.e. `B * max_step` of array traffic for a result the carry already holds.
The trip count is always static (`step < horizon`, from the int32 `horizon` in the bundle), never "while any episode is alive".
`returns_of` is the only place the `(P, R)` layout is assumed, and `evaluate` means over axis 1 after it.

### CartPole (`cartpole.py`)
Classic Gym `CartPole-v1` with Euler integration: `dt=0.02`, `g=9.8`, `masscart=1.0`, `masspole=0.1`, half-pole `length=0.5`, `force_mag=10`, dead when `|x| > 2.4` or `|theta| > 12deg` (`= 0.2095 rad`), reset `U(-0.05, 0.05)^4`.
Action: "push right" iff `out[1] > out[0]` (a tie pushes left, matching `argmax` index 0) computed with `enp.split` + `enp.greater`, so no integer/bool control flow and no `argmax` dtype handling.
One `evoxc.while_loop` over `max_step` timesteps with carry `(obs (B,4), alive (B,), total (B,), step ())`.
Step body: `total += alive` **before** the transition (the failing step still scores, matching Gym returns), `alive = alive & ~( |x'| > 2.4 | |theta'| > 12deg )` on the transition's own `x'`/`theta'`, state frozen after death via `enp.where(enp.expand_dims(alive, 1), next_obs, obs)` — no `cond`, every carried dtype/shape constant across iterations.
A traced tensor has no `__getitem__`: broadcast with `enp.expand_dims`, split columns with `enp.split`, never `x[:, 0]`.
Because episodes are frozen, an individual's return is its survival time: `max_step` is the fitness ceiling, so `fitness_target` must be `<= max_step`.
NaN logits act as push-left inside the rollout (`enp.greater` is false against NaN) but a NaN census (`evoxc.isnan(reduce_sum(logits, axis=1))`) is accumulated into `total`, so that individual's fitness is NaN → `-inf`.

### Pendulum (`pendulum.py`)
Gym `Pendulum-v1`: `theta=0` is **upright**, so the cost is small exactly at the unstable equilibrium.
Per timestep, all on the pre-transition state:
`u = max_torque * tanh(logit)`; `cost = angle_normalize(theta)**2 + 0.1*theta_dot**2 + 0.001*u**2`; `theta_dot' = clip(theta_dot + (3g/(2l)*sin(theta) + 3/(m*l**2)*u) * dt, -max_speed, max_speed)` — the clip comes **before** the angle update; `theta' = theta + theta_dot' * dt`; `total -= cost`.
`gravity_term = 3*gravity/(2*length)` and `torque_term = 3/(mass*length**2)` are precomputed host-side into the bundle, so the body never divides; the host oracle must use the same precomputed float32 values.
`angle_normalize(x) = x - two_pi * floor((x + pi) / two_pi)`, the floor-mod spelling of Gym's `((x + pi) % (2*pi)) - pi` (see Notes for why `evoxc.remainder` is not usable).
Reset: one `U(-1, 1)` block of shape `(B, 2)`, `theta = raw_theta * pi`, `theta_dot = raw_speed * max_speed`, recombined with `enp.concatenate(axis=1)`.
No termination and no alive mask: every episode is scored over the whole horizon, so the return is in `[-max_step * max_cost, 0]` (per-step cost is at most `pi**2 + 0.1*max_speed**2 + 0.001*max_torque**2 = 16.2736`), and `fitness_target` may only be `<= 0`.
Carry `(obs (B,2), total (B,), step ())`; `_advance` returns the cost already reshaped to `(B,)`, because `(B,) - (B,1)` would broadcast into a `(B, B)` matrix.
NaN: `tanh(NaN) = NaN` → `0.001*u**2` is NaN → `cost` is NaN → the accumulator is NaN → `finalize_fitness` gives `-inf`; verified empirically, so there is no NaN census.

### MountainCarContinuous (`mountaincar.py`)
Gym `MountainCarContinuous-v0`, exact Gym ordering on the pre-transition state:
`a = tanh(logit)` (the **pre-clip** action, which is what Gym charges the penalty on); `force = clip(a, -1, 1)` (a no-op because `|tanh| <= 1`, kept for faithfulness); `velocity' = velocity + force*power - gravity*cos(3*position)`; `velocity' = clip(velocity', -max_speed, max_speed)`; `position' = clip(position + velocity', min_position, max_position)`; `velocity' = 0` where `position' == min_position and velocity' < 0` (exact, because `clip` returns the bound verbatim); `done = position' >= goal_position and velocity' >= goal_velocity`; `reward = (100 if done else 0) - 0.1*a**2`.
Reset: one `U(-1, 1)` draw of shape `(B, 1)` scaled to `U(-0.6, -0.4)` (centre `-0.5`, half-width `0.1`, derived from the module literals `_RESET_LOW`/`_RESET_HIGH`) plus a zero velocity column.
**Termination differs from CartPole**: `done` is a *success*, the terminal transition pays the `100` bonus once and the `0.1*a**2` penalty is charged on it too, and afterwards the state is frozen (`enp.where(enp.expand_dims(alive, 1), next_obs, obs)`) **and no further reward is accumulated** — so the accumulator is masked by the **pre-step** alive flag, because the frozen post-goal state still satisfies `done` and an unmasked bonus would be re-awarded every remaining timestep.
Return bounds `[-(0.1*max_step), 100]`, hence the `fitness_target > 100` rejection; the honest ceiling is lower (~89–93 for a solving policy, since `0.1*a**2` accrues for the ~106 steps the climb takes).
Per-step traced ops are count-shaped, not element-shaped: keep `done` and the squared action rank-1 (`enp.reshape`) so no `(B,1)` operand meets the `(B,)` accumulator.
NaN: `tanh(NaN)` → the `0.1*a**2` penalty is NaN → the live step's reward is NaN → `-inf` through `finalize_fitness`; `enp.clip` is `clip(x, lo, hi) = minimum(maximum(x, lo), hi)`, and both ops propagate NaN, so the clip does not fold the poison away.

## Constraints
- Every arithmetic constant is an in-trace `enp.full((), value)` (float32) built **once per rollout, outside the loop** in the environment's `NamedTuple` bundle (plus the int32 `one_step`/`horizon`); a Python float literal would become f64 on the stablehlo path and break iree type unification, and `enp.full` inside the body would pay one Constant op per timestep.
- `evoxc.remainder` is **not** portable (floormod on numpy, fmod on iree); `angle_normalize` is written with `evoxc.floor`, which is bit-identical on both backends. `enp` has neither `floor` nor `remainder`.
- Use non-negative axes (`axis=1`) in `split`/`concatenate`, and broadcast rank-1 masks with `enp.expand_dims`; a traced tensor has no `__getitem__`.
- `enp.clip(x, lo, hi) = minimum(maximum(x, lo), hi)` accepts symbolic bounds, so no Python literal is needed for a bound.
- `_constants()` is built **once per rollout, outside the loop**: `enp.full` is a graph Constant op and the reference interpreter pays per traced op.
- Physics algebra is fixed op-for-op in all three: precomputing reciprocals (`1/total_mass`, `1/(2*l)`, …) or reassociating the float32 arithmetic changes which step a chaotic trajectory dies or reaches the goal on, so returns would stop matching the host-numpy oracles in `test/unit/problem/`.
- Control flow makes CartPole and MountainCarContinuous numpy / iree-llvm-cpu / xla only (TVM rejects `while`/`if`/gather/scatter) and the failure must surface as `evoxc.BackendError`, never a fallback. Pendulum is a `while_loop` too, so the same applies.
- `show` is host-side only (`evoxc.build` + `evoxc.run` on the key it is given), never called from a trace and never called with a schedule of more than one individual.
- `returns_of` is the only place the `(P, R)` layout is assumed; adding an environment means a step helper + a `while_loop` + the same finish step, in one file per environment.
- Files <= ~1000 lines (`cartpole.py` 639, `mountaincar.py` 693, `pendulum.py` 587, `_common.py` 165).

## Measured cost
Rollout component only (`evaluate` traced alone, no genome forward, toy policy — `1.5*sin(theta) + 0.5*theta_dot` for Pendulum, `3*position + 2*velocity + 1` for MountainCarContinuous, `(tanh(theta) - x, 2*theta + 3*theta_dot)` for CartPole), one `evoxc.build` then 7 `evoxc.run`, build excluded, **best median of 3 fresh processes**; evoxc 0.1.0, numpy 2.5.4, iree-base-compiler 3.11 (llvm-cpu), one 20-core box — the Pendulum/MountainCar rows on CPython 3.12, the CartPole rows on CPython 3.14 (an earlier session; the same box on 3.12 reads ≈7 % lower).
The box drifts ±30 % between processes, so the ratios are the load-bearing numbers and single-run medians can move by 2× under contention.
`R` is `repeat_times`; the `build` column is the one-off per static shape.

| env | pop | max_step | R | numpy ms | iree ms | ratio | build numpy / iree s |
| --- | --- | --- | --- | --- | --- | --- | --- |
| CartPole | 100 | 200 | 1 | 69.6 | 8.2 | 8.5× | 0.06 / 0.87 |
| CartPole | 1000 | 200 | 1 | 79.5 | 29.2 | 2.7× | 0.06 / 1.24 |
| Pendulum | 100 | 200 | 1 | 45.8 | 5.4 | 8.5× | 0.011 / 0.65 |
| Pendulum | 1000 | 200 | 1 | 49.9 | 14.9 | 3.4× | 0.012 / 0.66 |
| Pendulum | 100 | 500 | 1 | 116.9 | 13.2 | 8.9× | 0.012 / 0.65 |
| MountainCar | 100 | 200 | 1 | 59.6 | 6.0 | 9.9× | 0.014 / 0.69 |
| MountainCar | 1000 | 200 | 1 | 63.8 | 13.5 | 4.7× | 0.014 / 0.68 |
| MountainCar | 100 | 999 | 1 | 299.2 | 29.9 | 10.0× | 0.013 / 0.68 |
| MountainCar | 1000 | 999 | 1 | 318.9 | 80.8 | 4.0× | 0.019 / 0.69 |

Both new environments are cheap enough that the numpy interpreter's fixed per-op cost, not the element count, sets the time: 10× the population costs ≈1.07–1.09× (Pendulum/MountainCar at 200 steps) and 10× the horizon costs ≈2.55× (Pendulum 200 → 500) / ≈5.0× (MountainCar 200 → 999).
Their dynamics bodies are leaner than CartPole's 54 (the executed body is the prepared forward query plus the dynamics — see the per-timestep op budget below): Pendulum is 32 real ops (33 with the block terminator, ~29 without the toy policy's own 3) and MountainCar is 37 real ops (38 with the terminator), because neither carries an alive mask in the Pendulum case and MountainCar folds the goal test into the transition it already split.
iree is 3.4–10.0× faster than the numpy interpreter on both components, with a ≈0.65–0.70 s one-off build per shape.

### Per-timestep op budget (numpy interpreter, `P=100`, `R=1`)
The interpreter's fixed per-op cost sets the time, so the executed op count is the cost model: one `cartpole_p100` generation (`max_step=200`) is 31,604 op dispatches ≈ 250–277 ms on this box (7.9–8.8 µs/op), and it is exactly linear in the horizon — `ops = 3,804 + 139 × max_step` (measured at `max_step` 50/100/200/400; no term is quadratic).
At the committed benchmark protocol (150 generations, 3 repeats, seeds 42/43/44) `cartpole_p100` measures **305.7 ms per generation (260.9–311.6)**, 111 MiB peak RSS, 0.46 s build; the pre-hoist tree measures 380.8 (359.9–406.7) at the same protocol in the same session, and a single-step pair on the two trees reads 42,549 → 31,604 dispatches at the same 8.7 µs/op — 1.25× on the medians and 1.34× on that pair, with identical fitness.
Of the 139 ops per timestep, ~80 are the forward pass (the level loop's 76, the level `while` itself and the three NaN-seed constant/broadcast ops `forward_prepared` still emits per query), 5 are the two loop conditions plus the output read-out, and 54 are this node's CartPole force/dynamics/reward/carry update.
The rollout body region therefore executes 60 direct ops per timestep (56 data-dependent, 3 per-query constants, the block terminator), not 114: a rollout prepares its policy query once with `_common.prepared_query(act_fn, batch)` before the loop, so the genome's per-pass operands are derived once per generation instead of once per timestep.
`prepared_query` returns `act_fn.prepare(batch)` for the pipeline's `_NetworkActFn` and passes a genome-free `act_fn` (a unit suite's policy, a user kernel) through unchanged, which is what keeps `evaluate`/`show` working with a plain callable.
Before that hoist the same 58-op prelude (`_padding_safe_attrs`, `_hit_conn_attrs`' `(P,N,K,CA)` gather, `_slot_index`, `_output_row_mask`, `_depth_limit`, the NaN seed) was re-derived inside every timestep of every environment — 58 of Pendulum's 91 and of MountainCar's 98 executed body ops; the hoist took `cartpole_p100` from 42,549 to 31,604 ops and 364 to 249 ms (1.46×) with bit-identical fitness, and Pendulum/MountainCar from 19,615/13,590 to 16,370/10,345 ops (137→103 ms, 100→81 ms).
`repeat_times > 1` is the tiled-sample case of the evaluate path: the prepared pass is derived once for `B = P * R` rows while the schedule and the payload stay at `P`, so the genome's sample expansion (`genome/window.py::by_samples`, `NodeStepMixin._sample_count`) is paid once per generation rather than once per timestep. At `R=1` (`samples == 1`) that expansion is the identity and emits no op at all, which is why `R=1` and `R=4` now cost the same op count: `cartpole_p100` at `R=4` is 31,643 dispatches against 50,349 when the expansion ran per timestep (417.6 vs 587.3 ms on one step pair, identical fitness), i.e. 139 ops per timestep instead of 233.
Op-count attribution of one `cartpole_p100` generation: 27,802 ops (88 %) execute inside the rollout region and 3,802 (12 %) outside it (transform, speciate, tell and the graph entry); leaf kernel bodies are ≈42 % of the wall time and interpreter per-op work ≈58 %.
Comparability: `xor_p100` is 3,922 ops / ≈37 ms because one generation runs **one** population forward pass at `P*S=400` rows, while CartPole runs **200** sequential forward passes of ~80 ops each plus 54 dynamics ops per step — the ~10× is the 200 policy queries, not a per-gene or per-connection blow-up.

Pipeline level (`pop_batch_size` unset, one generation per `step`, numpy backend, CPython 3.12): `step` median, the same `step` with `problem.evaluate` monkeypatched to zeros, the difference (the evaluate share) and this node's rollout share of it.
Every `step` row below predates the prepared-query hoist (they measure a generation whose prelude is re-derived per timestep), so the CartPole rows are ≈1.25× their current value; the committed-protocol after numbers are in the paragraph above.

| env | pop | max_step | R | step ms | no-evaluate ms | evaluate share ms | rollout share of it | build s |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| CartPole | 100 | 200 | 1 | 440.9 | 50.1 | 391 | 18 % | 0.52 |
| CartPole | 1000 | 200 | 1 | 1848.9 | 599.8 | 1249 | 6 % | 1.15 |
| CartPole | 100 | 500 | 1 | 1016.9 | 49.1 | 968 | 18 % | 0.52 |
| CartPole | 1000 | 500 | 1 | 3803.9 | 603.1 | 3201 | 6 % | 1.15 |
| CartPole | 100 | 500 | 4 | 1866.7 | 52.8 | 1814 | 10 % | 0.52 |
| Pendulum (hidden 6) | 100 | 200 | 1 | 492.0 | 26.7 | 465 | 9.9 % | 0.278 |
| MountainCar | 100 | 200 | 1 | 377.8 | 36.3 | 342 | 17.5 % | 0.285 |

The genome `transform` + `forward` inside `act_fn` is 80–90 % of the evaluate share, exactly as for CartPole, so a continuous-control generation is genome-forward-bound, not rollout-bound.
A Pendulum generation with a **single-layer** genome instead of the 6-node hidden layer costs ≈228 ms at `P=100/max_step=200` (measured in the same session), i.e. the hidden layer roughly doubles the genome forward — worth it for the fitness it buys (see Notes).

Numeric agreement (`P=100`, one key, same toy policies): iree vs numpy per-individual fitness differs by `max_rel` 1.2e-6 (Pendulum 200 steps), 4.1e-7 (MountainCar 200) and 1.8e-6 (MountainCar 999) — float32 rounding only, no structural divergence.
Against a host-numpy **float32** transcription of the documented recurrence fed the graph's own initial states, both environments are bit-identical (`max_abs = 0`) at `max_step` 60, 200 and 999; against a **float64** Gym transcription the same runs differ by `max_rel` 1.2e-6 (Pendulum 200 steps) and 9.9e-7 (MountainCar 999 steps).
Cross-backend fitness is therefore reproducible to float32 rounding but never bitwise; no test may assert cross-backend trajectory equality.

## Notes for Agents
- **`evoxc.remainder` is not backend-consistent**: the numpy backend implements floor-mod (numpy `remainder`), while `stablehlo.remainder` — and hence `iree` — is `fmod` (sign of the dividend). A pendulum `angle_normalize` written with it wrapped `-3pi` to `-3pi` instead of `-pi` on iree and inflated the cost by 2.4× while the numpy side looked perfect. The floor form `x - two_pi*floor((x + pi)/two_pi)` is bit-identical on both. This is an `evoxc` doc/implementation mismatch (`evoxc/ops/elementwise.py` documents "sign of the dividend" but numpy does floormod) — out of scope here, worth an upstream report.
- **MountainCarContinuous's reward is deceptive**: `a -> 0` (any small-weight policy) returns exactly `0` (no bonus, no penalty), an aggressive constant policy returns `-0.1*max_step` (`-99.9` at 999, `-20` at 200), and only a bang-bang energy-pumping policy reaches the goal. Measured at `P=8`, `R=1`: `a = sign(velocity)` returns **90.8** (min 89.1) at **both** 999 and 200 steps — the horizon does not change a solving return, which also proves the `100` bonus is paid exactly once, while no-control returns `0` and push-right/push-left return `-0.1*max_step`; the environment therefore discriminates policies at 200 steps. NEAT nonetheless converges to the idle policy: `pop=100`, `species_size=10`, 12–20 generations, `max_step` 200/300/999, with and without `init_hidden_layers=(6,)`, all ended at `-0.007 … -0.0` (never the goal bonus). That is a property of the task, not of the kernel — pick a target near `-0.5` for a demo that terminates, or budget far more generations for the bonus.
- **Pendulum needs a hidden layer to learn quickly**: `pop=100`, `species_size=10`, `max_step=200`, `init_hidden_layers=()` plateaus at `-527` (best of 40 generations, seed 42), while `init_hidden_layers=(6,)` reaches `-238.8` at generation 8 of 20 (3.9 s wall). The 6-node layer costs ≈2× the genome forward.
- Recommended demo configurations, measured end to end on `backend="numpy"`, seed 42, `NEAT(pop_size=100, species_size=10, survival_threshold=0.1)` with `DefaultGenome(num_inputs=2, num_outputs=1, node_gene=BiasNode(tanh, sum))` and no `output_transform`:
  - Pendulum: `init_hidden_layers=(6,)`, `Pendulum(max_step=200, fitness_target=-300)`, `generation_limit=20` → target at **generation 8**, best `-238.84`, **3.93 s** wall (build 0.26 s).
  - MountainCarContinuous: `init_hidden_layers=()`, `MountainCarContinuous(max_step=200, fitness_target=-0.5)`, `generation_limit=10` → target at **generation 1**, best `-0.0067`, **0.53 s** wall (build 0.28 s), i.e. the idle policy; see the deception note above for why the goal bonus is out of reach at this budget.
- A NaN arriving **after** an episode has terminated is dropped by `MountainCarContinuous` (the accumulator is masked by the pre-step flag, and accumulation genuinely stops); `Pendulum` never terminates, so every NaN it sees lands. A NaN present while the episode is live always poisons it to `-inf`.
- `MountainCarContinuous.show` reports "goal not reached" as `return <= 0` rather than `return >= 100 - eps`: the `0.1*a**2` penalty accrues for the whole climb, so a solving episode returns ≈89, never ≈100.
- CartPole specifics worth not re-deriving: the `_loop_body` insertion order is load-bearing (`_nan_rows` selects the NaN into `total` **before** `alive` is updated, so a poison arriving on the step an episode dies still lands); `_terminated` stays rank-1 and is kept for host-side tests even though `_advance` tests the components it already has; `evoxc.scan` is **not** export-blocked on iree for this kernel (the old "rank-3 scatter" note applied to a rank-3 stacked output, which this kernel never had), `while_loop` is simply the cheaper form because it drops the stack; on `backend="numpy"` keep `max_step` small (cost is ≈0.35 ms per step per 100 individuals, linear in the horizon and nearly flat in the population).
- `_common.py` validators take the **owner name first** (`require_positive_int("Pendulum", "max_step", value)`) so the message shape matches `CartPole`'s original text exactly, which is why all three environments share them; `cartpole.py` passes `"CartPole"` and its suite pins the resulting `<Owner>: <name> must be ...` strings verbatim.
- This node's reference oracle for both new environments is a float32 host transcription (in `test/unit/problem/`, owned by the test agent), the same policy the CartPole suite uses; ordering decisions that a test can break: the Pendulum clip happens before the angle update, the MountainCar penalty uses the **pre-clip** action, the min-position velocity reset uses exact float equality, and the bonus is masked by the pre-step alive flag.
- `Pipeline` cannot build on iree today for reasons owned by `algorithm`/`genome`/`common` (rank-3 scatter on axis 0, a `uint32` constant, and `derive_schedule`'s `flat_scatter`), so the iree numbers above are component measurements and `backend="iree"` on a full pipeline must keep failing loudly.
- `_common.to_numpy` duplicates `common/tools.to_numpy`: the latter duck-types a `.numpy()` method (returning an `ndarray` unchanged), the former checks `isinstance(value, evoxc.Tensor)`. Both are host-only (the `show` report path) and all three environments use the `_common` one, while `problem/func_fit/func_fit.py` imports `common.to_numpy`; consolidating the pair is a pointer for the `common` owner, not a change to make in this node.

## Routing Table
| Path | Responsibility |
|---|---|
| `./cartpole.py` | Batched CartPole dynamics, fixed-horizon `while_loop` rollout with a carried return, survival-time fitness, NaN-poison accumulator, host-side report. |
| `./pendulum.py` | Batched Pendulum swing-up dynamics (`u = max_torque*tanh(logit)`, floor-mod angle wrap), fixed-horizon rollout, negative-cumulative-cost fitness, host-side report. |
| `./mountaincar.py` | Batched MountainCarContinuous dynamics in Gym order, goal-triggered termination with a one-time bonus masked by the pre-step alive flag, host-side report. |
| `./_common.py` | Host-side helpers shared by all three environments: `to_numpy`, the three owner-prefixed validators, the traced `returns_of` layout helper and the traced `prepared_query` that prepares a rollout's policy pass once. |
