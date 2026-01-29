"""
MO-NEAT XOR 快速测试脚本。
用于快速验证MO-NEAT是否正常工作。
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import time
import warnings
import jax
import jax.numpy as jnp
import numpy as np

from tensorneat import genome, problem
from tensorneat.genome import OriginNode, OriginConn
from tensorneat.common import ACT, State

from mo_neat import MONEAT
from nsga2_utils import compute_pareto_front


def main():
    print("=" * 60)
    print("MO-NEAT XOR Quick Test")
    print("=" * 60)
    
    # 配置 - 快速测试参数
    POP_SIZE = 1000
    GENERATION_LIMIT = 100
    FITNESS_TARGET = -1e-4
    SEED = 42
    
    np.random.seed(SEED)
    
    alg = MONEAT(
        pop_size=POP_SIZE,
        elite_ratio=0.1,
        tournament_size=2,
        genome=genome.DefaultGenome(
            node_gene=OriginNode(),
            conn_gene=OriginConn(),
            num_inputs=2,
            num_outputs=1,
            max_nodes=10,
            max_conns=30,
            output_transform=ACT.sigmoid,
        ),
    )
    
    prob = problem.XOR()
    
    print("\nInitializing...")
    state = State()
    state = state.register(randkey=jax.random.PRNGKey(SEED))
    state = alg.setup(state)
    state = prob.setup(state)
    
    print("Compiling...")
    
    def step(state):
        randkey_, randkey = jax.random.split(state.randkey)
        pop = alg.ask(state)
        pop_transformed = jax.vmap(alg.transform, in_axes=(None, 0))(state, pop)
        keys = jax.random.split(randkey_, POP_SIZE)
        fitnesses = jax.vmap(prob.evaluate, in_axes=(None, 0, None, 0))(
            state, keys, alg.forward, pop_transformed
        )
        fitnesses = jnp.where(jnp.isnan(fitnesses), -jnp.inf, fitnesses)
        previous_pop = alg.ask(state)
        state_new = alg.tell(state, fitnesses)
        return state_new.update(randkey=randkey), previous_pop, fitnesses
    
    tic = time.time()
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        compiled_step = jax.jit(step).lower(state).compile()
    print(f"Compile time: {time.time() - tic:.2f}s\n")
    
    best_fitness = float("-inf")
    best_genome = None
    
    for gen in range(GENERATION_LIMIT):
        gen_start = time.time()
        state, previous_pop, fitnesses = compiled_step(state)
        fitnesses_np = jax.device_get(fitnesses)
        
        max_idx = np.argmax(fitnesses_np)
        if fitnesses_np[max_idx] > best_fitness:
            best_fitness = fitnesses_np[max_idx]
            best_genome = (previous_pop[0][max_idx], previous_pop[1][max_idx])
        
        valid_fitnesses = fitnesses_np[~np.isinf(fitnesses_np)]
        complexity = jax.device_get(
            alg.calculate_complexity(state.pop_nodes, state.pop_conns)
        )
        cost_time = time.time() - gen_start
        
        if (gen + 1) % 10 == 0 or gen == 0:
            print(
                f"Gen {int(state.generation):3d} | "
                f"Time: {cost_time * 1000:.0f}ms | "
                f"Fitness: max={np.max(valid_fitnesses):.6f}, mean={np.mean(valid_fitnesses):.6f} | "
                f"Complexity: min={np.min(complexity)}, mean={np.mean(complexity):.1f}"
            )
        
        if best_fitness >= FITNESS_TARGET:
            print(f"\n*** Fitness target reached at gen {int(state.generation)}! ***")
            break
    
    # 最终结果
    print("\n" + "=" * 60)
    print("Final Results")
    print("=" * 60)
    
    final_complexity = jax.device_get(
        alg.calculate_complexity(state.pop_nodes, state.pop_conns)
    )
    final_objectives = np.stack([fitnesses_np, -final_complexity], axis=1)
    is_front = jax.device_get(compute_pareto_front(jnp.array(final_objectives)))
    
    # 过滤掉无效解（fitness为-inf的）
    valid_front = is_front & ~np.isinf(fitnesses_np)
    
    front_fitness = fitnesses_np[valid_front]
    front_complexity = final_complexity[valid_front]
    
    print(f"\nValid Pareto front size: {len(front_fitness)}")
    print(f"Best fitness overall: {best_fitness:.6f}")
    
    if len(front_fitness) > 0:
        print("\nPareto front solutions (sorted by fitness):")
        sort_idx = np.argsort(-front_fitness)[:10]
        for i, idx in enumerate(sort_idx):
            print(f"  {i+1}. Fitness: {front_fitness[idx]:.6f}, Complexity: {front_complexity[idx]}")
    
    if best_genome is not None:
        print("\nBest genome:")
        print(alg.genome.repr(state, *best_genome))
    
    return state


if __name__ == "__main__":
    main()
