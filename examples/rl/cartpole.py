"""Balance the CartPole with a NEAT-evolved policy.

Run with::

    python examples/rl/cartpole.py

Backend note: the rollout is a fixed-horizon scan, so evaluation cost grows
linearly with ``max_step`` and the numpy reference interpreter is slow at the
default ``max_step=500`` (roughly 180 ms per evaluation at a population of
100). For real runs prefer ``backend="iree"`` (or ``"xla"``) with the full
horizon; this example keeps ``backend="numpy"`` and a reduced ``max_step`` so
it finishes in seconds without any optional extra installed.
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

# Small budget: a generation costs seconds on the reference backend.
pipeline = Pipeline(
    algorithm=neat,
    problem=cartpole,
    seed=42,
    generation_limit=20,
    fitness_target=200,
    backend="numpy",
)

if __name__ == "__main__":
    state = pipeline.setup()
    state, best = pipeline.auto_run(state)
    pipeline.show(state, best)
