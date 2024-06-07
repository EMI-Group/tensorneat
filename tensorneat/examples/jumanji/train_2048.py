import jax, jax.numpy as jnp

from pipeline import Pipeline
from algorithm.neat import *
from algorithm.neat.gene.node.default_without_response import NodeGeneWithoutResponse
from problem.rl_env.jumanji.jumanji_2048 import Jumanji_2048
from utils import Act, Agg


def rot_li(li):
    return li[1:] + [li[0]]


def rot_boards(board):
    def rot(a, _):
        a = jnp.rot90(a)
        return a, a  # carry, y

    # carry, np.stack(ys)
    _, boards = jax.lax.scan(rot, board, jnp.arange(4, dtype=jnp.int32))
    return boards


direction = ["up", "right", "down", "left"]
lr_flip_direction = ["up", "left", "down", "right"]

directions = []
lr_flip_directions = []
for _ in range(4):
    direction = rot_li(direction)
    lr_flip_direction = rot_li(lr_flip_direction)
    directions.append(direction.copy())
    lr_flip_directions.append(lr_flip_direction.copy())

full_directions = directions + lr_flip_directions


def action_policy(forward_func, obs):
    board = obs.reshape(4, 4)
    lr_flip_board = jnp.fliplr(board)

    boards = rot_boards(board)
    lr_flip_boards = rot_boards(lr_flip_board)
    # stack
    full_boards = jnp.concatenate([boards, lr_flip_boards], axis=0)
    scores = jax.vmap(forward_func)(full_boards.reshape(8, -1))
    total_score = {"up": 0, "right": 0, "down": 0, "left": 0}
    for i in range(8):
        dire = full_directions[i]
        for j in range(4):
            total_score[dire[j]] += scores[i, j]

    return jnp.array(
        [
            total_score["up"],
            total_score["right"],
            total_score["down"],
            total_score["left"],
        ]
    )


if __name__ == "__main__":
    pipeline = Pipeline(
        algorithm=NEAT(
            species=DefaultSpecies(
                genome=DefaultGenome(
                    num_inputs=16,
                    num_outputs=4,
                    max_nodes=100,
                    max_conns=1000,
                    node_gene=NodeGeneWithoutResponse(
                        activation_default=Act.sigmoid,
                        activation_options=(
                            Act.sigmoid,
                            Act.relu,
                            Act.tanh,
                            Act.identity,
                        ),
                        aggregation_default=Agg.sum,
                        aggregation_options=(Agg.sum,),
                        activation_replace_rate=0.02,
                        aggregation_replace_rate=0.02,
                        bias_mutate_rate=0.03,
                        bias_init_std=0.5,
                        bias_mutate_power=0.2,
                        bias_replace_rate=0.01,
                    ),
                    conn_gene=DefaultConnGene(
                        weight_mutate_rate=0.015,
                        weight_replace_rate=0.003,
                        weight_mutate_power=0.5,
                    ),
                    mutation=DefaultMutation(node_add=0.001, conn_add=0.002),
                ),
                pop_size=1000,
                species_size=5,
                survival_threshold=0.1,
                max_stagnation=7,
                genome_elitism=3,
                compatibility_threshold=1.2,
            ),
        ),
        problem=Jumanji_2048(
            max_step=10000,
            repeat_times=10,
            guarantee_invalid_action=True,
            action_policy=action_policy,
        ),
        generation_limit=1000,
        fitness_target=13000,
        save_path="2048.npz",
    )

    # initialize state
    state = pipeline.setup()
    # print(state)
    # run until terminate
    state, best = pipeline.auto_run(state)
