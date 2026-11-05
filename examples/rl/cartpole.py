"""Balance the CartPole with a NEAT-evolved policy.

Run with::

    python examples/rl/cartpole.py

Backend note: the rollout is a fixed-horizon loop, so evaluation cost grows
linearly with ``max_step``. On this CPU the numpy reference interpreter spends
roughly 0.38 s per generation at ``pop_size=100`` with the default
``max_step=500`` (measured), and about 0.21–0.28 s per generation for the budget
below (``pop_size=200``, ``max_step=200``). This script therefore keeps
``backend="numpy"`` and a reduced ``max_step`` so it finishes in seconds without
any optional extra installed; a compiled backend pays a one-off compile per
static shape and, on CPU, is measured to be no faster than numpy for this
workload, so reach for it for memory, not for speed.

With ``seed=42`` the run below reaches ``fitness_target=200`` -- a full
200-step episode -- after a couple of generations.
"""

from tensorneat import Pipeline, algorithm, genome, problem
from tensorneat.common import ACT, AGG

# Backend switch:
#   backend="numpy" — reference interpreter, always available, no extras.
#   backend="iree" / backend="xla" — opt-in extras; the graph is compiled
#     once per shape and reused for every generation.
# An unsupported backend raises instead of silently falling back, so the same
# script runs unchanged on whatever backend you have installed.

# Two output logits: the problem pushes right iff out[1] > out[0].
# No output_transform here — an argmax-style transform would collapse the two
# logits into one value and destroy the ordering the environment needs.
neat = algorithm.NEAT(
    pop_size=200,
    species_size=10,
    survival_threshold=0.1,
    genome=genome.DefaultGenome(
        num_inputs=4,  # (x, x_dot, theta, theta_dot)
        num_outputs=2,  # two action logits
        init_hidden_layers=(),
        node_gene=genome.BiasNode(
            activation_options=ACT.tanh,
            aggregation_options=AGG.sum,
        ),
    ),
)

# A frozen episode's return is its survival time, so max_step is the fitness
# ceiling: the problem raises TensorNEATError if fitness_target > max_step.
cartpole = problem.CartPole(max_step=200, repeat_times=1, fitness_target=200)

# Small budget: about a quarter second per generation on the reference backend.
pipeline = Pipeline(
    algorithm=neat,
    problem=cartpole,
    seed=42,
    generation_limit=20,
    fitness_target=200,
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
