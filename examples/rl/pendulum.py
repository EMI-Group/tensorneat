"""Swing the Pendulum upright with a NEAT-evolved continuous-control policy.

Run with::

    python examples/rl/pendulum.py

The environment is ``problem.Pendulum`` -- Gym's ``Pendulum-v1`` dynamics
written as evoxc ops, no simulator installed. ``theta = 0`` is upright, so a
good policy keeps the pole near the unstable equilibrium; the return is the
negative cumulative cost over a fixed 200-step horizon and is therefore never
positive (``fitness_target`` may only be ``<= 0``).

Backend note: the rollout is a fixed-horizon loop, so evaluation cost grows
with ``max_step``; at ``pop_size=100`` / ``max_step=200`` a generation costs
0.23–0.40 s on this CPU (measured) and the 0.28–0.30 s build is one-off, so the
run below (5 generations) takes 2.7–2.9 s wall including interpreter startup
(measured). All these figures come from the numpy reference interpreter, which
needs no optional extra, so this script keeps ``backend="numpy"``. A compiled
backend pays a one-off
compile per static shape and, on CPU, is measured faster for the raw rollout
but slower end to end for this workload; reach for it for portability and
memory, not for speed here.

With ``seed=42`` the run below reaches ``fitness_target=-300`` (mean return at
or above -300, i.e. a cost below 300) at generation 5 of the 20 allowed: the
per-generation log tops out at -271.30. ``show`` builds its own one-episode
executable on ``state.randkey`` and reports -595.12 for that genome, a
different measurement rather than a contradiction. The 6-node hidden layer
matters: the same budget without ``init_hidden_layers`` plateaus around -527.
"""

from tensorneat import Pipeline, algorithm, genome, problem
from tensorneat.common import ACT, AGG

# Backend switch:
#   backend="numpy" — reference interpreter, always available, no extras.
#   backend="iree" / backend="xla" — opt-in extras; the graph is compiled
#     once per shape and reused for every generation.
# An unsupported backend raises instead of silently falling back, so the same
# script runs unchanged on whatever backend you have installed.

# One output logit: a continuous torque. Hidden layer included because a
# single tanh layer cannot express the swing-up; see the docstring.
neat = algorithm.NEAT(
    pop_size=100,
    species_size=10,
    survival_threshold=0.1,
    genome=genome.DefaultGenome(
        num_inputs=2,  # (theta, theta_dot)
        num_outputs=1,  # one torque logit
        init_hidden_layers=(6,),
        node_gene=genome.BiasNode(
            activation_options=ACT.tanh,
            aggregation_options=AGG.sum,
        ),
    ),
)

# No output_transform here: the problem owns the squashing, u = max_torque *
# tanh(logit). A bounded transform would rescale the logit first and silently
# shrink the effective torque range.
pendulum = problem.Pendulum(max_step=200, fitness_target=-300)

# Every Pendulum return is -cost <= 0, so the target is a negative bound.
pipeline = Pipeline(
    algorithm=neat,
    problem=pendulum,
    seed=42,
    generation_limit=20,
    fitness_target=-300,
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
