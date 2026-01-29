"""
NSGA-II utilities for multi-objective optimization in JAX.
基于 QDax 库的实现，适配 TensorNEAT 的用法。
"""

import jax
import jax.numpy as jnp
from functools import partial


def compute_pareto_dominance(
    criteria_point: jax.Array, batch_of_criteria: jax.Array
) -> jax.Array:
    """
    判断一个点是否被其他点支配（使用最大化约定）。
    
    Args:
        criteria_point: 一个目标值向量 (num_criteria,)
        batch_of_criteria: 一批目标值向量 (num_points, num_criteria)
    
    Returns:
        bool: 如果该点被批次中的任意点支配，返回True
    """
    diff = jnp.subtract(batch_of_criteria, criteria_point)
    diff_greater_than_zero = jnp.any(diff > 0, axis=-1)
    diff_geq_than_zero = jnp.all(diff >= 0, axis=-1)
    return jnp.any(jnp.logical_and(diff_greater_than_zero, diff_geq_than_zero))


def compute_pareto_front(batch_of_criteria: jax.Array) -> jax.Array:
    """
    计算Pareto前沿，返回每个个体是否在前沿上。
    
    Args:
        batch_of_criteria: 一批目标值 (num_points, num_criteria)
    
    Returns:
        bool array: 每个点是否在Pareto前沿上
    """
    func = jax.vmap(lambda x: ~compute_pareto_dominance(x, batch_of_criteria))
    return func(batch_of_criteria)


def compute_masked_pareto_dominance(
    criteria_point: jax.Array, batch_of_criteria: jax.Array, mask: jax.Array
) -> jax.Array:
    """
    带mask的Pareto支配判断。
    
    Args:
        criteria_point: 目标值向量 (num_criteria,)
        batch_of_criteria: 一批目标值 (batch_size, num_criteria)
        mask: 掩码 (batch_size,), True表示该位置被屏蔽
    
    Returns:
        bool: 是否被支配
    """
    diff = jnp.subtract(batch_of_criteria, criteria_point)
    # 对于被屏蔽的点，设置为负值使其不影响支配判断
    neutral_values = -jnp.ones_like(diff)
    diff = jax.vmap(lambda x1, x2: jnp.where(mask, x1, x2), in_axes=(1, 1), out_axes=1)(
        neutral_values, diff
    )
    diff_greater_than_zero = jnp.any(diff > 0, axis=-1)
    diff_geq_than_zero = jnp.all(diff >= 0, axis=-1)
    return jnp.any(jnp.logical_and(diff_greater_than_zero, diff_geq_than_zero))


def compute_masked_pareto_front(batch_of_criteria: jax.Array, mask: jax.Array) -> jax.Array:
    """
    带mask的Pareto前沿计算。
    
    Args:
        batch_of_criteria: 目标值 (num_points, num_criteria)
        mask: 掩码 (num_points,), True表示该位置被屏蔽
    
    Returns:
        bool array: 未屏蔽且在前沿上的点
    """
    func = jax.vmap(
        lambda x: ~compute_masked_pareto_dominance(x, batch_of_criteria, mask)
    )
    return func(batch_of_criteria) * ~mask


def compute_crowding_distances(fitnesses: jax.Array, mask: jax.Array) -> jax.Array:
    """
    计算拥挤度距离。
    
    拥挤度距离是目标空间中的曼哈顿距离，用于在同一前沿内排序个体。
    
    Args:
        fitnesses: 适应度值 (num_solutions, num_objectives)
        mask: 掩码 (num_solutions,), True表示该位置被屏蔽（不参与计算）
    
    Returns:
        拥挤度距离数组 (num_solutions,)
    """
    num_solutions = fitnesses.shape[0]
    num_objectives = fitnesses.shape[1]
    
    # 处理边界情况
    def small_pop_case():
        return jnp.where(mask, -jnp.inf, jnp.inf)
    
    def normal_case():
        # 对被屏蔽的解决方案，设置一个很大的值使其排在后面
        mask_dist = jnp.column_stack([mask] * num_objectives)
        score_amplitude = jnp.max(fitnesses, axis=0) - jnp.min(fitnesses, axis=0)
        dist_fitnesses = (
            fitnesses + 3 * score_amplitude * jnp.ones_like(fitnesses) * mask_dist
        )
        
        # 对每个目标维度排序
        sorted_index = jnp.argsort(dist_fitnesses, axis=0)
        srt_fitnesses = fitnesses[sorted_index, jnp.arange(num_objectives)]
        
        # 计算每个目标的范围
        norm = jnp.max(srt_fitnesses, axis=0) - jnp.min(srt_fitnesses, axis=0)
        norm = jnp.where(norm == 0, 1.0, norm)  # 避免除以0
        
        # 计算距离
        dists = jnp.vstack(
            [srt_fitnesses, jnp.full(num_objectives, jnp.inf)]
        ) - jnp.vstack([jnp.full(num_objectives, -jnp.inf), srt_fitnesses])
        
        dist_to_last = dists[:-1] / norm
        dist_to_next = dists[1:] / norm
        
        # 恢复原始顺序并求和
        j = jnp.argsort(sorted_index, axis=0)
        crowding_distances = (
            jnp.sum(
                (
                    dist_to_last[j, jnp.arange(num_objectives)]
                    + dist_to_next[j, jnp.arange(num_objectives)]
                ),
                axis=1,
            )
            / num_objectives
        )
        
        # 被屏蔽的解设为-inf
        crowding_distances = jnp.where(mask, -jnp.inf, crowding_distances)
        return crowding_distances
    
    return jax.lax.cond(num_solutions <= 2, small_pop_case, normal_case)


def nsga2_selection(
    fitnesses: jax.Array,
    pop_size: int,
) -> jax.Array:
    """
    NSGA-II选择算子：基于非支配排序和拥挤度距离选择下一代。
    
    Args:
        fitnesses: 所有个体的适应度 (total_pop, num_objectives)
        pop_size: 要选择的个体数量
        
    Returns:
        被选中个体的索引 (pop_size,)
    """
    num_candidates = fitnesses.shape[0]
    
    def compute_current_front(val):
        """计算当前的Pareto前沿"""
        to_keep_index, _ = val
        front_index = compute_masked_pareto_front(fitnesses, mask=to_keep_index)
        to_keep_index = to_keep_index + front_index
        return to_keep_index, front_index
    
    def condition_fn_1(val):
        """检查是否已选够个体"""
        to_keep_index, _ = val
        return jnp.sum(to_keep_index) < pop_size
    
    # 迭代计算连续的Pareto前沿，直到选够个体
    to_keep_index, front_index = jax.lax.while_loop(
        condition_fn_1,
        compute_current_front,
        (
            jnp.zeros(num_candidates, dtype=bool),
            jnp.zeros(num_candidates, dtype=bool),
        ),
    )
    
    # 移除最后一个前沿的索引（可能超出需要的数量）
    new_index = jnp.arange(start=1, stop=num_candidates + 1) * to_keep_index
    new_index = new_index * (~front_index)
    to_keep_index_without_last = new_index > 0
    
    # 计算最后一个前沿的拥挤度
    crowding_distances = compute_crowding_distances(fitnesses, ~front_index)
    crowding_distances = crowding_distances * front_index  # 只考虑最后一个前沿
    highest_dist = jnp.argsort(crowding_distances)  # 升序
    
    def add_to_front(val):
        """从最后一个前沿中按拥挤度添加个体"""
        front_index_add, num = val
        front_index_add = front_index_add.at[highest_dist[-num - 1]].set(True)
        num = num + 1
        return front_index_add, num
    
    def condition_fn_2(val):
        """检查是否已选够个体"""
        front_index_add, _ = val
        return jnp.sum(to_keep_index_without_last + front_index_add) < pop_size
    
    # 从最后一个前沿按拥挤度添加个体
    front_index_add, _ = jax.lax.while_loop(
        condition_fn_2,
        add_to_front,
        (jnp.zeros(num_candidates, dtype=bool), 0),
    )
    
    # 合并所有选中的个体
    final_keep_index = to_keep_index_without_last + front_index_add
    
    # 转换为索引
    indices = jnp.arange(num_candidates)
    selected_indices = jnp.where(final_keep_index, indices, num_candidates)
    selected_indices = jnp.sort(selected_indices)[:pop_size]
    
    return selected_indices


def compute_ranks_and_crowding(fitnesses: jax.Array) -> tuple[jax.Array, jax.Array]:
    """
    计算所有个体的Pareto等级和拥挤度距离。
    
    Args:
        fitnesses: 适应度值 (pop_size, num_objectives)
    
    Returns:
        ranks: Pareto等级 (pop_size,), 0是最好的（第一前沿）
        crowding: 拥挤度距离 (pop_size,)
    """
    pop_size = fitnesses.shape[0]
    
    def body_fn(carry, _):
        ranks, assigned_mask, current_rank = carry
        
        # 计算当前前沿
        front_mask = compute_masked_pareto_front(fitnesses, mask=assigned_mask)
        
        # 更新ranks
        ranks = jnp.where(front_mask, current_rank, ranks)
        
        # 更新已分配的mask
        assigned_mask = assigned_mask | front_mask
        
        return (ranks, assigned_mask, current_rank + 1), None
    
    # 初始化
    init_ranks = jnp.full(pop_size, pop_size, dtype=jnp.int32)  # 默认最差等级
    init_mask = jnp.zeros(pop_size, dtype=bool)
    
    # 迭代计算每一层的等级（最多pop_size层）
    (ranks, _, _), _ = jax.lax.scan(
        body_fn, 
        (init_ranks, init_mask, 0), 
        None, 
        length=pop_size
    )
    
    # 计算拥挤度（对每个等级分别计算）
    crowding = jnp.zeros(pop_size)
    
    def compute_crowding_for_rank(carry, rank):
        crowding = carry
        rank_mask = ranks == rank
        
        # 只有当该等级有个体时才计算
        def has_individuals():
            cd = compute_crowding_distances(fitnesses, ~rank_mask)
            return jnp.where(rank_mask, cd, crowding)
        
        def no_individuals():
            return crowding
        
        crowding = jax.lax.cond(
            jnp.any(rank_mask),
            has_individuals,
            no_individuals
        )
        return crowding, None
    
    crowding, _ = jax.lax.scan(
        compute_crowding_for_rank,
        crowding,
        jnp.arange(pop_size)
    )
    
    return ranks, crowding


def tournament_selection_mo(
    key: jax.Array,
    ranks: jax.Array,
    crowding: jax.Array,
    num_selections: int,
    tournament_size: int = 2,
) -> jax.Array:
    """
    多目标锦标赛选择。
    
    比较规则：
    1. 优先选择Pareto等级低的（更接近前沿）
    2. 等级相同时选择拥挤度高的（更稀疏区域）
    
    Args:
        key: 随机数key
        ranks: Pareto等级 (pop_size,)
        crowding: 拥挤度距离 (pop_size,)
        num_selections: 选择数量
        tournament_size: 锦标赛大小
    
    Returns:
        被选中个体的索引 (num_selections,)
    """
    pop_size = ranks.shape[0]
    
    def single_tournament(key):
        # 随机选择tournament_size个个体
        candidates = jax.random.choice(key, pop_size, shape=(tournament_size,), replace=False)
        
        # 获取它们的rank和crowding
        cand_ranks = ranks[candidates]
        cand_crowding = crowding[candidates]
        
        # 选择最好的：先按rank（越小越好），再按crowding（越大越好）
        # 组合成一个可比较的值：rank * large_number - crowding
        large_num = 1e10
        scores = cand_ranks * large_num - cand_crowding
        best_idx = jnp.argmin(scores)
        
        return candidates[best_idx]
    
    keys = jax.random.split(key, num_selections)
    return jax.vmap(single_tournament)(keys)
