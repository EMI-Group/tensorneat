"""
MO-NEAT XOR 最终测试脚本。
使用更高的变异概率来更好地解决XOR问题。
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
from tensorneat.genome.operations.mutation import DefaultMutation
from tensorneat.common import ACT, State

from mo_neat import MONEAT
from nsga2_utils import compute_pareto_front


def main():
    print("=" * 60)
    print("MO-NEAT XOR Final Test")
    print("Objectives: 1. Fitness (maximize) 2. Complexity (minimize)")
    print("=" * 60)
    
    # 配置
    POP_SIZE = 5000  # 较小种群用于快速测试
    GENERATION_LIMIT = 1500
    FITNESS_TARGET = -1e-4
    SEED = 42
    
    np.random.seed(SEED)
    
    # 使用自定义的变异参数，增加添加节点的概率
    custom_mutation = DefaultMutation(
        conn_add=0.3,
        conn_delete=0.1,
        node_add=0.2,  # 增加添加节点的概率
        node_delete=0.05,
    )
    
    alg = MONEAT(
        pop_size=POP_SIZE,
        elite_ratio=0.05,
        tournament_size=3,
        genome=genome.DefaultGenome(
            node_gene=OriginNode(),
            conn_gene=OriginConn(),
            num_inputs=2,
            num_outputs=1,
            max_nodes=15,
            max_conns=50,
            output_transform=ACT.sigmoid,
            mutation=custom_mutation,
        ),
    )
    
    prob = problem.XOR()
    
    print(f"\nConfiguration:")
    print(f"  Population size: {POP_SIZE}")
    print(f"  Generations: {GENERATION_LIMIT}")
    print(f"  Mutation rates: node_add=0.2, conn_add=0.3")
    print()
    
    print("Initializing...")
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
    best_complexity = float("inf")
    
    # 记录历史
    history = {
        'generation': [],
        'best_fitness': [],
        'mean_fitness': [],
        'min_complexity': [],
        'mean_complexity': [],
        'pareto_size': [],
    }
    
    print("Starting evolution...")
    print("-" * 80)
    
    total_start = time.time()
    
    for gen in range(GENERATION_LIMIT):
        gen_start = time.time()
        state, previous_pop, fitnesses = compiled_step(state)
        fitnesses_np = jax.device_get(fitnesses)
        
        # 更新最佳
        max_idx = np.argmax(fitnesses_np)
        if fitnesses_np[max_idx] > best_fitness:
            best_fitness = fitnesses_np[max_idx]
            best_genome = (previous_pop[0][max_idx], previous_pop[1][max_idx])
            
            # 计算最佳个体的复杂度
            best_nodes = jax.device_get(best_genome[0])
            best_conns = jax.device_get(best_genome[1])
            best_complexity = (~np.isnan(best_nodes[:, 0])).sum() + (~np.isnan(best_conns[:, 0])).sum()
        
        valid_fitnesses = fitnesses_np[~np.isinf(fitnesses_np)]
        complexity = jax.device_get(
            alg.calculate_complexity(state.pop_nodes, state.pop_conns)
        )
        cost_time = time.time() - gen_start
        
        # 计算Pareto前沿大小
        objectives = np.stack([fitnesses_np, -complexity], axis=1)
        is_front = jax.device_get(compute_pareto_front(jnp.array(objectives)))
        valid_front = is_front & ~np.isinf(fitnesses_np)
        pareto_size = np.sum(valid_front)
        
        # 记录历史
        history['generation'].append(int(state.generation))
        history['best_fitness'].append(best_fitness)
        history['mean_fitness'].append(np.mean(valid_fitnesses))
        history['min_complexity'].append(np.min(complexity))
        history['mean_complexity'].append(np.mean(complexity))
        history['pareto_size'].append(pareto_size)
        
        if (gen + 1) % 20 == 0 or gen == 0:
            print(
                f"Gen {int(state.generation):3d} | "
                f"Time: {cost_time * 1000:.0f}ms | "
                f"Valid: {len(valid_fitnesses):5d} | "
                f"Best: {best_fitness:.6f} | "
                f"Mean: {np.mean(valid_fitnesses):.6f} | "
                f"Complexity: {np.min(complexity)}-{np.mean(complexity):.1f} | "
                f"Pareto: {pareto_size}"
            )
        
        if best_fitness >= FITNESS_TARGET:
            print(f"\n*** Fitness target reached at gen {int(state.generation)}! ***")
            break
    
    total_time = time.time() - total_start
    
    # 最终结果
    print("\n" + "=" * 60)
    print("Final Results")
    print("=" * 60)
    
    print(f"\nTotal runtime: {total_time:.2f}s ({total_time/60:.2f} minutes)")
    print(f"Generations completed: {int(state.generation)}")
    print(f"Time per generation: {total_time/int(state.generation)*1000:.1f}ms")
    
    # 最终Pareto前沿
    final_complexity = jax.device_get(
        alg.calculate_complexity(state.pop_nodes, state.pop_conns)
    )
    final_objectives = np.stack([fitnesses_np, -final_complexity], axis=1)
    is_front = jax.device_get(compute_pareto_front(jnp.array(final_objectives)))
    valid_front = is_front & ~np.isinf(fitnesses_np)
    
    front_fitness = fitnesses_np[valid_front]
    front_complexity = final_complexity[valid_front]
    
    print(f"\n--- Pareto Front Analysis ---")
    print(f"Valid Pareto front size: {len(front_fitness)}")
    
    if len(front_fitness) > 0:
        print(f"\nFitness range: [{np.min(front_fitness):.6f}, {np.max(front_fitness):.6f}]")
        print(f"Complexity range: [{np.min(front_complexity)}, {np.max(front_complexity)}]")
        
        print("\nTop 10 solutions from Pareto front:")
        print("-" * 50)
        sort_idx = np.argsort(-front_fitness)[:10]
        for i, idx in enumerate(sort_idx):
            print(f"  {i+1}. Fitness: {front_fitness[idx]:.6f}, Complexity: {front_complexity[idx]}")
        
        # 找到"膝部点"
        if len(front_fitness) > 2:
            norm_fitness = (front_fitness - front_fitness.min()) / (front_fitness.max() - front_fitness.min() + 1e-8)
            norm_complexity = (front_complexity.max() - front_complexity) / (front_complexity.max() - front_complexity.min() + 1e-8)
            combined_score = norm_fitness + norm_complexity
            knee_idx = np.argmax(combined_score)
            print(f"\nKnee point: Fitness={front_fitness[knee_idx]:.6f}, Complexity={front_complexity[knee_idx]}")
    
    print(f"\n--- Best Individual ---")
    print(f"Best fitness: {best_fitness:.6f}")
    print(f"Best complexity: {best_complexity}")
    
    if best_genome is not None:
        print("\nNetwork structure:")
        print(alg.genome.repr(state, *best_genome))
    
    # 保存历史数据
    history_file = 'mo_neat_history.npz'
    np.savez(
        history_file,
        generation=history['generation'],
        best_fitness=history['best_fitness'],
        mean_fitness=history['mean_fitness'],
        min_complexity=history['min_complexity'],
        mean_complexity=history['mean_complexity'],
        pareto_size=history['pareto_size'],
    )
    print(f"\nHistory saved to {history_file}")
    
    return state, history


if __name__ == "__main__":
    main()
