"""
MO-NEAT XOR 测试脚本。
测试多目标NEAT在XOR问题上的表现。

目标：
1. 最大化 fitness (负MSE误差)
2. 最小化网络复杂度 (节点数 + 连接数)
"""

import sys
import os

# 添加src路径
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
    print("MO-NEAT XOR Test")
    print("Objectives: 1. Fitness (maximize) 2. Complexity (minimize)")
    print("=" * 60)
    
    # 配置
    POP_SIZE = 5000  # 增大种群以获得更好的探索
    GENERATION_LIMIT = 300
    FITNESS_TARGET = -1e-5
    SEED = 42
    
    # 设置随机种子
    np.random.seed(SEED)
    
    # 创建MO-NEAT算法
    alg = MONEAT(
        pop_size=POP_SIZE,
        elite_ratio=0.05,  # 减少精英比例以增加探索
        tournament_size=3,  # 增加锦标赛大小以增加选择压力
        genome=genome.DefaultGenome(
            node_gene=OriginNode(),
            conn_gene=OriginConn(),
            num_inputs=2,
            num_outputs=1,
            max_nodes=15,
            max_conns=50,
            output_transform=ACT.sigmoid,
        ),
    )
    
    # 创建问题
    prob = problem.XOR()
    
    # 初始化状态
    print("\nInitializing...")
    state = State()
    state = state.register(randkey=jax.random.PRNGKey(SEED))
    state = alg.setup(state)
    state = prob.setup(state)
    
    # 编译step函数
    print("Compiling...")
    
    def step(state):
        randkey_, randkey = jax.random.split(state.randkey)
        
        pop = alg.ask(state)
        
        # 转换种群
        pop_transformed = jax.vmap(alg.transform, in_axes=(None, 0))(
            state, pop
        )
        
        # 评估
        keys = jax.random.split(randkey_, POP_SIZE)
        fitnesses = jax.vmap(prob.evaluate, in_axes=(None, 0, None, 0))(
            state, keys, alg.forward, pop_transformed
        )
        
        # 处理NaN
        fitnesses = jnp.where(jnp.isnan(fitnesses), -jnp.inf, fitnesses)
        
        previous_pop = alg.ask(state)
        state_new = alg.tell(state, fitnesses)
        
        return state_new.update(randkey=randkey), previous_pop, fitnesses
    
    tic = time.time()
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        compiled_step = jax.jit(step).lower(state).compile()
    print(f"Compile finished, cost time: {time.time() - tic:.2f}s\n")
    
    # 主循环
    best_fitness = float("-inf")
    best_genome = None
    
    # 记录Pareto前沿历史
    pareto_history = []
    
    for gen in range(GENERATION_LIMIT):
        gen_start = time.time()
        
        state, previous_pop, fitnesses = compiled_step(state)
        fitnesses_np = jax.device_get(fitnesses)
        
        # 更新最佳
        max_idx = np.argmax(fitnesses_np)
        if fitnesses_np[max_idx] > best_fitness:
            best_fitness = fitnesses_np[max_idx]
            best_genome = (previous_pop[0][max_idx], previous_pop[1][max_idx])
        
        # 计算统计信息
        valid_fitnesses = fitnesses_np[~np.isinf(fitnesses_np)]
        cost_time = time.time() - gen_start
        
        # 计算Pareto前沿
        complexity = jax.device_get(
            alg.calculate_complexity(state.pop_nodes, state.pop_conns)
        )
        objectives = np.stack([fitnesses_np, -complexity], axis=1)
        
        # 只在valid fitness上计算
        valid_mask = ~np.isinf(fitnesses_np)
        
        print(
            f"Gen {int(state.generation):3d} | "
            f"Time: {cost_time * 1000:.1f}ms | "
            f"Valid: {len(valid_fitnesses):4d} | "
            f"Fitness max: {np.max(valid_fitnesses):.6f}, "
            f"mean: {np.mean(valid_fitnesses):.6f} | "
            f"Complexity min: {np.min(complexity)}, "
            f"mean: {np.mean(complexity):.1f}"
        )
        
        # 显示详细信息 (每20代)
        if (gen + 1) % 20 == 0:
            alg.show_details(state, fitnesses_np)
        
        # 记录Pareto前沿 (每50代)
        if (gen + 1) % 50 == 0:
            # 获取前沿
            objectives_jnp = jnp.array(objectives)
            is_front = jax.device_get(compute_pareto_front(objectives_jnp))
            front_fitness = fitnesses_np[is_front]
            front_complexity = complexity[is_front]
            pareto_history.append({
                'gen': int(state.generation),
                'fitness': front_fitness.copy(),
                'complexity': front_complexity.copy(),
            })
        
        # 检查终止条件
        if best_fitness >= FITNESS_TARGET:
            print(f"\nFitness target reached! Best fitness: {best_fitness:.6f}")
            break
    
    if int(state.generation) >= GENERATION_LIMIT:
        print(f"\nGeneration limit reached! Best fitness: {best_fitness:.6f}")
    
    # 显示最终结果
    print("\n" + "=" * 60)
    print("Final Results")
    print("=" * 60)
    
    # 最终Pareto前沿
    final_complexity = jax.device_get(
        alg.calculate_complexity(state.pop_nodes, state.pop_conns)
    )
    final_objectives = np.stack([fitnesses_np, -final_complexity], axis=1)
    final_objectives_jnp = jnp.array(final_objectives)
    is_front = jax.device_get(compute_pareto_front(final_objectives_jnp))
    
    front_indices = np.where(is_front)[0]
    front_fitness = fitnesses_np[is_front]
    front_complexity = final_complexity[is_front]
    
    print(f"\nPareto front size: {len(front_indices)}")
    print("\nSample solutions from Pareto front:")
    print("-" * 50)
    
    # 按fitness排序显示
    sort_idx = np.argsort(-front_fitness)
    for i, idx in enumerate(sort_idx[:10]):  # 显示前10个
        print(f"  {i+1}. Fitness: {front_fitness[idx]:.6f}, Complexity: {front_complexity[idx]}")
    
    if len(sort_idx) > 10:
        print(f"  ... and {len(sort_idx) - 10} more solutions")
    
    # 显示最佳fitness个体
    print("\n" + "-" * 50)
    best_idx = np.argmax(front_fitness)
    print(f"Best fitness solution: fitness={front_fitness[best_idx]:.6f}, complexity={front_complexity[best_idx]}")
    
    # 显示最简单个体
    min_complexity_idx = np.argmin(front_complexity)
    print(f"Simplest solution: fitness={front_fitness[min_complexity_idx]:.6f}, complexity={front_complexity[min_complexity_idx]}")
    
    # 显示"膝部"解决方案（trade-off点）
    # 简单方法：标准化后选择综合得分最高的
    if len(front_fitness) > 2:
        norm_fitness = (front_fitness - front_fitness.min()) / (front_fitness.max() - front_fitness.min() + 1e-8)
        norm_complexity = (front_complexity.max() - front_complexity) / (front_complexity.max() - front_complexity.min() + 1e-8)
        combined_score = norm_fitness + norm_complexity
        knee_idx = np.argmax(combined_score)
        print(f"Knee point solution: fitness={front_fitness[knee_idx]:.6f}, complexity={front_complexity[knee_idx]}")
    
    print("\n" + "=" * 60)
    
    # 可视化最佳个体
    if best_genome is not None:
        print("\nBest genome representation:")
        print(alg.genome.repr(state, *best_genome))
    
    return state, pareto_history


if __name__ == "__main__":
    main()
