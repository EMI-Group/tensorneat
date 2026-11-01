# Intent

Task/fitness definitions for TensorNEAT: a problem wraps an environment or dataset and turns a transformed genome ("params") into one scalar fitness that the NEAT algorithm maximizes.
Every problem is a `StatefulBaseClass` (`src/tensorneat/common/stateful_class.py`) so it can register immutable values into the shared `State` pytree.
Two families: function fitting (`func_fit/`, dataset known at construction) and reinforcement learning (`rl/`, environment stepped inside a JAX loop).

# API Surface (`base.py: BaseProblem`)

- `jitable` class attribute (default `None`) — must be set to `True` by every subclass; `Pipeline.__init__` asserts `problem.jitable` (`pipeline.py:33`), so all shipped problems set `jitable = True`.
- `evaluate(self, state, randkey, act_func, params)` — return a scalar fitness, maximized; implement in every subclass (`Functions`/`RLEnv` do).
- `act_func` is `algorithm.forward`, i.e. `Genome.forward(state, transformed_params, inputs) -> outputs` (`genome/base.py:107`, `genome/default.py:68`); it is passed as a static (non-vmapped) argument.
- Pipeline call site: `jax.vmap(problem.evaluate, in_axes=(None, 0, None, 0))(state, keys, self.algorithm.forward, pop_transformed)` (`pipeline.py:119`) — the param-pytree and per-individual randkey are the vmapped axes.
- `setup(self, state=State()) -> State` — inherited no-op from `StatefulBaseClass.setup`; override to `state.register(...)`; it runs after `algorithm.setup` (`pipeline.py:86-87`) so `state.randkey` is available.
- `input_shape` / `output_shape` properties — `input_shape[-1]` must equal `algorithm.num_inputs` (asserted in `Pipeline.__init__`).
- `show(self, state, randkey, act_func, params, *args, **kwargs)` — non-jitted, human-facing visualization/printout.
- `show_details(self, state, randkey, act_func, pop_params, *args, **kwargs)` — inherited no-op; called from `Pipeline.auto_run` only when `show_problem_details=True`.
- `State` is an immutable registered pytree (`common/state.py`): use `register`/`update`/`remove`, attribute assignment raises `AttributeError("State is immutable")`.

# Shipped Subclasses

- `func_fit/func_fit.py: FuncFit(BaseProblem)` — abstract dataset problem; `evaluate` does `vmap(act_func, in_axes=(None, None, 0))` over `self.inputs`, computes `mse|rmse|mae|mape` loss and returns `-loss`; abstract `inputs`, `targets`, `input_shape`, `output_shape`.
- `func_fit/custom.py: CustomFuncFit(FuncFit)` — builds a dataset from a user `func` over `low_bounds`/`upper_bounds` (`method="sample"` with `num_samples`, or `method="grid"` with `step_size`); helpers `generate_dataset`, `cartesian_product`.
- `func_fit/xor.py: XOR(FuncFit)` — 4 rows `[[0,0],[0,1],[1,0],[1,1]]`, targets `[[0],[1],[1],[0]]`, shapes `(4,2)/(4,1)`.
- `func_fit/xor3d.py: XOR3d(FuncFit)` — 8 rows of 3-bit inputs, parity targets, shapes `(8,3)/(8,1)`.
- `rl/rl_jit.py: RLEnv(BaseProblem)` — jitted rollout engine; all RL subclasses implement only `env_step(randkey, env_state, action)` and `env_reset(randkey)`; `evaluate` averages `repeat_times` episodes via `vmap(self._evaluate_once, ...)`.
- `rl/gymnax.py: GymNaxEnv(RLEnv)` — `env_name` must be in `gymnax.registered_envs`; wraps `env.step(key, state, action, env_params)` / `env.reset(key, env_params)`; `show` raises `NotImplementedError`.
- `rl/brax.py: BraxEnv(RLEnv)` — `envs.create(env_name, backend="generalized")`; sizes from `env.observation_size`/`env.action_size`; `show` renders a trajectory to rgb_array/gif/mp4.
- `rl/mujoco_playground.py: MujocoEnv(RLEnv)` — `registry.load(env_name=env_name)` (default `"SwimmerSwimmer6"`); unwraps dict observations by the `"state"` key; `show` supports gif/mp4 only.

# Constraints and Conventions

- `evaluate` (and everything it calls) executes inside `jax.jit(Pipeline.step)` and inside `lax.while_loop`: no Python control flow on traced values, no printing, no `np.*` conversion, no in-place mutation — use `.at[i].set(v)` and functional returns.
- Fitness is maximized; function fitting returns `-loss`, RL returns mean episodic reward; the pipeline maps `NaN` fitness to `-inf` (`pipeline.py:150`).
- `evaluate` must be vmappable over both the randkey axis and the param-pytree axis, so it must not carry Python-level batching state between calls.
- Problems are pure containers of config: environment handles are created once in `__init__` and closed over by the jitted functions.
- `setup` must return a new `State` (`state.register(...)`), never mutate the incoming one.
- Shape convention is inconsistent by family: `FuncFit.input_shape/output_shape` include the leading batch dimension (e.g. `(4, 2)`), while `RLEnv` shapes are the per-step observation/action shape (e.g. `(4,)`).

# Dependencies

- Hot path (performance critical): pure JAX — `jax.vmap`, `jax.lax.while_loop` over timesteps, `jax.random.split` (`rl/rl_jit.py:132-177`).
- Bookkeeping (host side, un-jitted): dataset construction with NumPy (`func_fit/custom.py`), observation-normalization sampling with one `jax.jit(vmap(...))` in `RLEnv.setup`, printed logs.
- Optional leaf deps, all lazily imported inside functions (never at module import): `gymnax` in `rl/gymnax.py:6`, `brax` in `rl/brax.py:9` and `brax.io.image` in `rl/brax.py:48`/`rl/mujoco_playground.py:65`, `mujoco_playground` in `rl/mujoco_playground.py:10`, `imageio` inside both `show` methods.
- Only `import jax` / `jax.numpy` are top-level in this package, so importing `tensorneat.problem` needs no simulator installed; constructing `GymNaxEnv`/`BraxEnv`/`MujocoEnv` does.
- `flax`, `optax`, `jaxopt` are declared in `pyproject.toml` but are not imported anywhere under `src/tensorneat/`.

# Config Knobs (`RLEnv.__init__`)

- `max_step=1000` — episode horizon; also the padded length of the arrays returned when `record_episode=True`.
- `repeat_times=1` — number of independent episodes averaged in `evaluate` (stochasticity reduction, NOT action repeat / frame skip).
- `action_policy=None` — callable `(randkey, forward_func, obs) -> action`; when `None`, the raw network output is used as the action.
- `obs_normalization=False` — when `True` requires `sample_policy=(randkey, obs) -> action` and `sample_episodes>0`; `setup` runs random episodes, registers `problem_obs_mean`/`problem_obs_std` into `State`, and `norm_obs(state, obs)` (`rl/rl_jit.py:208`) applies `(obs-mean)/(std+1e-6)` inside the rollout.
- Env parameters come from the library, not from this package: `gymnax.make(env_name)` returns `(env, env_params)`, brax/mujoco-playground use `envs.create` / `registry.load` defaults; there is no local env registry.

# Known Issues / Gotchas

- `RLEnv._evaluate_once` uses `jax.lax.while_loop` (cond `~done & (count < max_step)`), not `lax.scan`; episode recording pre-fills arrays with `NaN` so unfinished steps stay distinguishable.
- Pytree observation spaces are unsupported in `MujocoEnv`: any obs dict lacking a `"state"` key raises `ImportError` (raised) from `env_step`/`env_reset`.
- `record_episode` is a private argument of `_evaluate_once`, not a constructor flag; the `record_episode=True` examples in `test/test_record_episode.*` use an older `problem.rl_env` import path and no longer match this API.
- `BraxEnv.show` / `MujocoEnv.show` pass the same un-split `randkey` into every `jit_step` call, so stochastic envs see identical randomness per step.
- `CustomFuncFit.generate_dataset` uses the global `np.random` (not a JAX key), so its dataset is only reproducible because `Pipeline.__init__` calls `np.random.seed(self.seed)`.
- `XOR`/`XOR3d` return NumPy arrays from `inputs`/`targets`, which works because they are closed-over constants under `vmap`/`jit`.

# Routing Table

- `./src/tensorneat/problem/func_fit` — supervised function-fitting problems (`FuncFit`, `CustomFuncFit`, `XOR`, `XOR3d`).
- `./src/tensorneat/problem/rl` — RL rollout engine and simulator adapters (`RLEnv`, `GymNaxEnv`, `BraxEnv`, `MujocoEnv`).
