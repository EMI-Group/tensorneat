"""
XOR with gradient descent only (no evolution).

Demonstrates that TensorNEAT can simultaneously optimize networks with
different topologies via gradient descent. The workflow:
  1. Initialize a population of identical networks.
  2. Mutate each network with a different random seed (20 rounds),
     so they develop diverse topologies.
  3. Optimize all of them in parallel using pure gradient descent.
"""

import jax, jax.numpy as jnp
import numpy as np
from tensorneat import algorithm, genome, problem
from tensorneat.common import ACT, State
from tensorneat.genome import DefaultMutation

POPSIZE = 1000
MUTATION_ROUNDS = 20
GRAD_STEPS = 500
GRAD_LR = 0.5
LOG_EVERY = 10

# ── 1. Setup ──────────────────────────────────────────────────────────

state = State(randkey=jax.random.key(42))

neat = algorithm.NEAT(
    pop_size=POPSIZE,
    species_size=20,
    survival_threshold=0.01,
    genome=genome.DefaultGenome(
        num_inputs=3,
        num_outputs=1,
        max_nodes=50,
        max_conns=100,
        init_hidden_layers=(3,),
        output_transform=ACT.sigmoid,
        mutation=DefaultMutation(
            conn_add=0.5,
            conn_delete=0.1,
            node_add=0.3,
            node_delete=0.05,
        ),
    ),
)

prob = problem.XOR3d()
state = neat.setup(state)
state = prob.setup(state)

g = neat.genome

# ── 2. Mutate to create diverse topologies ────────────────────────────

pop_nodes, pop_conns = neat.ask(state)

jit_batch_mutate = jax.jit(
    jax.vmap(g.execute_mutation, in_axes=(None, 0, 0, 0, 0, 0))
)

randkey = state.randkey
for i in range(MUTATION_ROUNDS):
    k1, k2, randkey = jax.random.split(randkey, 3)
    mutate_keys = jax.random.split(k1, POPSIZE)

    max_node_key = jnp.nanmax(pop_nodes[:, :, 0])
    new_node_keys = jnp.arange(POPSIZE) + max_node_key + 1
    new_conn_markers = jnp.full((POPSIZE, 3), 0)

    pop_nodes, pop_conns = jit_batch_mutate(
        state, mutate_keys, pop_nodes, pop_conns, new_node_keys, new_conn_markers
    )

node_counts = jnp.sum(~jnp.isnan(pop_nodes[:, :, 0]), axis=1)
conn_counts = jnp.sum(~jnp.isnan(pop_conns[:, :, 0]), axis=1)
print(
    f"After {MUTATION_ROUNDS} mutation rounds: "
    f"nodes {int(jnp.min(node_counts))}~{int(jnp.max(node_counts))}, "
    f"conns {int(jnp.min(conn_counts))}~{int(jnp.max(conn_counts))}"
)

# ── 3. Gradient descent on diverse topologies ─────────────────────────

loss_fn = lambda preds: jnp.mean((preds - prob.targets) ** 2)


def grad_step(nodes, conns, state):
    loss, (grads_n, grads_c) = g.grad(state, nodes, conns, prob.inputs, loss_fn)
    return nodes - GRAD_LR * grads_n, conns - GRAD_LR * grads_c, loss


batch_grad_step = jax.jit(
    jax.vmap(grad_step, in_axes=(0, 0, None))
)

for step in range(1, GRAD_STEPS + 1):
    pop_nodes, pop_conns, losses = batch_grad_step(pop_nodes, pop_conns, state)

    if step % LOG_EVERY == 0 or step == 1:
        cpu_losses = jax.device_get(losses)
        valid_mask = np.isfinite(cpu_losses)
        best_loss = np.min(cpu_losses[valid_mask]) if valid_mask.any() else float("nan")
        mean_loss = np.mean(cpu_losses[valid_mask]) if valid_mask.any() else float("nan")
        valid = np.sum(valid_mask)
        print(
            f"Step {step:>4d} | "
            f"best_loss: {best_loss:.6f} | "
            f"mean_loss: {mean_loss:.6f} | "
            f"valid: {valid}"
        )

    if best_loss < 1e-6:
        print("Solved!")
        break
else:
    print("Step limit reached.")

# ── 4. Show result ───────────────────────────────────────────────────

cpu_losses = jax.device_get(losses)
best_idx = int(np.nanargmin(cpu_losses))
best = (pop_nodes[best_idx], pop_conns[best_idx])

print(f"\nBest individual (loss={cpu_losses[best_idx]:.6f}):")
print(g.repr(state, *best))

jit_batch_forward = jax.jit(jax.vmap(g.forward, in_axes=(None, None, 0)))
best_transformed = jax.jit(g.transform)(state, *best)
preds = jit_batch_forward(state, best_transformed, prob.inputs)
for x, y in zip(prob.inputs, preds):
    print(f"  input={jax.device_get(x)} -> output={jax.device_get(y)}")
