"""Drive the MountainCarContinuous problem with a NEAT-evolved policy.

Run with::

    python examples/rl/mountaincar.py

The environment is ``problem.MountainCarContinuous`` -- Gym's
``MountainCarContinuous-v0`` dynamics written as evoxc ops. The car starts at
rest in the valley and has to pump the slope to reach the goal at
``position >= 0.45``. This script keeps the Gym-faithful default horizon
(``max_step=999``), so the example also shows the default rather than a
trimmed demo budget.

Read the reward honestly before judging the number it prints: the return is
``100`` once at the goal minus ``0.1 * a**2`` per step. So (a) the idle policy
``a = 0`` returns exactly ``0.0`` -- no bonus, no penalty -- and (b) even a
solving episode nets only about ``+89``, because the penalty accrues over the
whole climb. A hand-written ``a = sign(velocity)`` bang-bang policy scores
``90.8`` here; "always push right" scores ``-99.9``. NEAT at this budget
converges to the *idle* optimum: the run below ends at mean return ``-0.00``
and ``show`` reports "goal not reached". That is a property of the task's
reward, not of the kernel, so this example demonstrates the integration
(environment + one-logit policy shape + pipeline) end to end; it does not
solve the control task, and it sets ``fitness_target=float("inf")`` on purpose
so the run goes the full generation budget instead of stopping on the first
idle individual (the pipeline's stop test is ``max fitness >= fitness_target``).

Backend note: the rollout is a fixed-horizon loop, so evaluation cost grows
linearly with ``max_step`` -- 999 steps at ``pop_size=100`` costs 0.63–1.24 s
per generation on this CPU (measured). That is why the budget below is 10
generations: about 10.4 s of generations, a 0.28 s build and ~1 s of
interpreter startup, 11.6–11.8 s wall in total, on the numpy reference
interpreter, which needs no optional extra. A compiled
backend pays a one-off compile per static shape and, on CPU, is measured no
faster end to end for this workload.
"""

from tensorneat import Pipeline, algorithm, genome, problem
from tensorneat.common import ACT, AGG

# Backend switch:
#   backend="numpy" — reference interpreter, always available, no extras.
#   backend="iree" / backend="xla" — opt-in extras; the graph is compiled
#     once per shape and reused for every generation.
# An unsupported backend raises instead of silently falling back, so the same
# script runs unchanged on whatever backend you have installed.

# One output logit: a continuous push. No hidden layer -- this budget is sized
# to show the integration, not to search for an energy-pumping policy.
neat = algorithm.NEAT(
    pop_size=100,
    species_size=10,
    survival_threshold=0.1,
    genome=genome.DefaultGenome(
        num_inputs=2,  # (position, velocity)
        num_outputs=1,  # one action logit
        init_hidden_layers=(),
        node_gene=genome.BiasNode(
            activation_options=ACT.tanh,
            aggregation_options=AGG.sum,
        ),
    ),
)

# No max_step override: the Gym-faithful 999-step horizon is the default.
# No output_transform either -- the problem owns the squashing, a = tanh(logit).
mountaincar = problem.MountainCarContinuous()

# The pipeline's stop test is `max fitness >= fitness_target`, so inf means
# "never stop early": the run always uses the whole generation_limit.
pipeline = Pipeline(
    algorithm=neat,
    problem=mountaincar,
    seed=42,
    generation_limit=10,
    fitness_target=float("inf"),
    backend="numpy",
)

if __name__ == "__main__":
    # Host-only: seeds the RNG key, registers backend/compile options, and
    # builds the initial population and the task tables. No graph is traced.
    state = pipeline.setup()
    # The evoxc graph is built by the first step() inside auto_run -- one-off
    # per static shape, then cached for every generation (it prints
    # "compile finished, cost time: ...s" once, before the generation log).
    state, best = pipeline.auto_run(state)
    pipeline.show(state, best)
