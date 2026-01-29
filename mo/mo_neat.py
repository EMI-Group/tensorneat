"""
Multi-Objective NEAT (MO-NEAT) 算法实现。
结合 NSGA-II 的非支配排序和 TensorNEAT 的神经拓扑演化。

目标：
1. Fitness (最大化)
2. Network complexity (最小化) - 节点数 + 连接数
"""

import jax
from jax import vmap, numpy as jnp
import numpy as np

from tensorneat.common import State
from tensorneat.genome import BaseGenome

from nsga2_utils import (
    compute_ranks_and_crowding,
    tournament_selection_mo,
    compute_pareto_front,
)


class MONEAT:
    """
    Multi-Objective NEAT 算法。
    
    使用 NSGA-II 的非支配排序来同时优化：
    - Fitness (任务性能)
    - 网络复杂度 (节点数 + 连接数，越小越好)
    """
    
    def __init__(
        self,
        genome: BaseGenome,
        pop_size: int,
        elite_ratio: float = 0.1,
        tournament_size: int = 2,
    ):
        """
        Args:
            genome: 基因组类
            pop_size: 种群大小
            elite_ratio: 精英比例（直接保留到下一代）
            tournament_size: 锦标赛大小
        """
        self.genome = genome
        self.pop_size = pop_size
        self.elite_ratio = elite_ratio
        self.num_elites = max(1, int(pop_size * elite_ratio))
        self.tournament_size = tournament_size
    
    def setup(self, state=State()):
        """初始化状态"""
        state = self.genome.setup(state)
        
        k1, randkey = jax.random.split(state.randkey, 2)
        
        # 初始化种群
        initialize_keys = jax.random.split(k1, self.pop_size)
        pop_nodes, pop_conns = vmap(self.genome.initialize, in_axes=(None, 0))(
            state, initialize_keys
        )
        
        state = state.register(
            pop_nodes=pop_nodes,
            pop_conns=pop_conns,
            generation=jnp.float32(0),
            # MO-NEAT specific state (all float32 for consistency)
            last_ranks=jnp.zeros(self.pop_size, dtype=jnp.float32),
            last_crowding=jnp.zeros(self.pop_size, dtype=jnp.float32),
            last_complexity=jnp.zeros(self.pop_size, dtype=jnp.float32),
        )
        
        return state.update(randkey=randkey)
    
    def ask(self, state):
        """返回当前种群"""
        return state.pop_nodes, state.pop_conns
    
    def calculate_complexity(self, pop_nodes, pop_conns):
        """
        计算每个个体的网络复杂度。
        复杂度 = 活跃节点数 + 活跃连接数
        
        Args:
            pop_nodes: (pop_size, max_nodes, node_attrs)
            pop_conns: (pop_size, max_conns, conn_attrs)
        
        Returns:
            complexity: (pop_size,)
        """
        # 使用非NaN计数来确定活跃的节点和连接
        # 假设第一个属性列是key/ID
        active_nodes = ~jnp.isnan(pop_nodes[:, :, 0])
        active_conns = ~jnp.isnan(pop_conns[:, :, 0])
        
        node_counts = jnp.sum(active_nodes, axis=1)
        conn_counts = jnp.sum(active_conns, axis=1)
        
        return node_counts + conn_counts
    
    def tell(self, state, fitness):
        """
        基于适应度和复杂度更新种群。
        
        Args:
            state: 当前状态
            fitness: 每个个体的任务适应度 (pop_size,)
        
        Returns:
            更新后的状态
        """
        state = state.update(generation=state.generation + 1)
        
        pop_nodes = state.pop_nodes
        pop_conns = state.pop_conns
        
        # 计算复杂度
        complexity = self.calculate_complexity(pop_nodes, pop_conns)
        
        # 构建多目标适应度矩阵
        # fitness 需要最大化，complexity 需要最小化
        # 为了统一使用最大化，将 complexity 取负
        objectives = jnp.stack([fitness, -complexity], axis=1)
        
        # 计算 Pareto 等级和拥挤度
        ranks, crowding = compute_ranks_and_crowding(objectives)
        
        # 保存用于显示（转换为float32以保持一致）
        state = state.update(
            last_ranks=ranks.astype(jnp.float32),
            last_crowding=crowding.astype(jnp.float32),
            last_complexity=complexity.astype(jnp.float32),
        )
        
        # 选择精英（基于rank和crowding排序）
        # 精英 = rank最小，crowding最大
        large_num = 1e10
        elite_scores = ranks * large_num - crowding
        elite_indices = jnp.argsort(elite_scores)[:self.num_elites]
        
        # 创建下一代
        state = self._create_next_generation(
            state, ranks, crowding, fitness, elite_indices
        )
        
        return state
    
    def _create_next_generation(self, state, ranks, crowding, fitness, elite_indices):
        """创建下一代种群"""
        
        # 找到下一个节点key
        all_nodes_keys = state.pop_nodes[:, :, 0]
        max_node_key = jnp.max(
            all_nodes_keys, where=~jnp.isnan(all_nodes_keys), initial=0
        )
        next_node_key = max_node_key + 1
        new_node_keys = jnp.arange(self.pop_size) + next_node_key
        
        # 连接历史标记（如果需要）
        if "historical_marker" in self.genome.conn_gene.fixed_attrs:
            all_conns_markers = vmap(
                self.genome.conn_gene.get_historical_marker, in_axes=(None, 0)
            )(state, state.pop_conns)
            
            max_conn_markers = jnp.max(
                all_conns_markers, where=~jnp.isnan(all_conns_markers), initial=0
            )
            next_conn_markers = max_conn_markers + 1
            new_conn_markers = (
                jnp.arange(self.pop_size * 3).reshape(self.pop_size, 3)
                + next_conn_markers
            )
        else:
            new_conn_markers = jnp.full((self.pop_size, 3), 0)
        
        # 准备随机数keys
        k1, k2, k3, randkey = jax.random.split(state.randkey, 4)
        
        # 使用锦标赛选择父代（为所有位置选择）
        winner_indices = tournament_selection_mo(
            k1, ranks, crowding, self.pop_size, self.tournament_size
        )
        loser_indices = tournament_selection_mo(
            k2, ranks, crowding, self.pop_size, self.tournament_size
        )
        
        # 确保 winner fitness >= loser fitness
        winner_fitness = fitness[winner_indices]
        loser_fitness = fitness[loser_indices]
        is_winner = winner_fitness >= loser_fitness
        
        final_winner = jnp.where(is_winner, winner_indices, loser_indices)
        final_loser = jnp.where(is_winner, loser_indices, winner_indices)
        
        wpn, wpc = state.pop_nodes[final_winner], state.pop_conns[final_winner]
        lpn, lpc = state.pop_nodes[final_loser], state.pop_conns[final_loser]
        
        # 随机数keys
        crossover_randkeys = jax.random.split(k3, self.pop_size)
        k4, randkey = jax.random.split(randkey)
        mutate_randkeys = jax.random.split(k4, self.pop_size)
        
        # 批量交叉
        n_nodes, n_conns = vmap(
            self.genome.execute_crossover, in_axes=(None, 0, 0, 0, 0, 0)
        )(state, crossover_randkeys, wpn, wpc, lpn, lpc)
        
        # 批量变异
        m_n_nodes, m_n_conns = vmap(
            self.genome.execute_mutation, in_axes=(None, 0, 0, 0, 0, 0)
        )(state, mutate_randkeys, n_nodes, n_conns, new_node_keys, new_conn_markers)
        
        # 获取精英的基因组
        elite_nodes = state.pop_nodes[elite_indices]
        elite_conns = state.pop_conns[elite_indices]
        
        # 构建最终种群：前num_elites个是精英，其余是变异后的后代
        # 使用动态切片来正确组合
        pop_nodes = m_n_nodes.at[:self.num_elites].set(elite_nodes)
        pop_conns = m_n_conns.at[:self.num_elites].set(elite_conns)
        
        return state.update(
            randkey=randkey,
            pop_nodes=pop_nodes,
            pop_conns=pop_conns,
        )
    
    def transform(self, state, individual):
        """转换个体为可执行形式"""
        nodes, conns = individual
        return self.genome.transform(state, nodes, conns)
    
    def forward(self, state, transformed, inputs):
        """前向传播"""
        return self.genome.forward(state, transformed, inputs)
    
    @property
    def num_inputs(self):
        return self.genome.num_inputs
    
    @property
    def num_outputs(self):
        return self.genome.num_outputs
    
    def show_details(self, state, fitness):
        """显示详细信息"""
        pop_nodes, pop_conns = jax.device_get([state.pop_nodes, state.pop_conns])
        nodes_cnt = (~np.isnan(pop_nodes[:, :, 0])).sum(axis=1)
        conns_cnt = (~np.isnan(pop_conns[:, :, 0])).sum(axis=1)
        
        complexity = nodes_cnt + conns_cnt
        
        # 获取保存的ranks和crowding
        ranks = jax.device_get(state.last_ranks)
        crowding = jax.device_get(state.last_crowding)
        
        # Pareto前沿统计
        front_0_mask = ranks == 0
        front_0_count = np.sum(front_0_mask)
        
        front_0_fitness = fitness[front_0_mask] if front_0_count > 0 else np.array([])
        front_0_complexity = complexity[front_0_mask] if front_0_count > 0 else np.array([])
        
        print(
            f"\tnode counts: max: {max(nodes_cnt)}, min: {min(nodes_cnt)}, mean: {np.mean(nodes_cnt):.2f}\n",
            f"\tconn counts: max: {max(conns_cnt)}, min: {min(conns_cnt)}, mean: {np.mean(conns_cnt):.2f}\n",
            f"\tcomplexity: max: {max(complexity)}, min: {min(complexity)}, mean: {np.mean(complexity):.2f}\n",
            f"\tPareto front size: {front_0_count}\n",
        )
        
        if front_0_count > 0:
            print(
                f"\tFront 0 fitness: max: {np.max(front_0_fitness):.4f}, min: {np.min(front_0_fitness):.4f}\n",
                f"\tFront 0 complexity: max: {np.max(front_0_complexity)}, min: {np.min(front_0_complexity)}\n",
            )
    
    def get_pareto_front(self, state, fitness):
        """
        获取当前Pareto前沿上的个体。
        
        Args:
            state: 当前状态
            fitness: 适应度值
        
        Returns:
            front_indices: 前沿上个体的索引
            front_fitness: 前沿上个体的fitness
            front_complexity: 前沿上个体的complexity
            front_individuals: (nodes, conns) 前沿上的个体
        """
        complexity = self.calculate_complexity(state.pop_nodes, state.pop_conns)
        objectives = jnp.stack([fitness, -complexity], axis=1)
        
        is_front = compute_pareto_front(objectives)
        
        front_indices = jnp.where(is_front, jnp.arange(self.pop_size), -1)
        front_indices = front_indices[front_indices >= 0]
        
        return (
            front_indices,
            fitness[is_front],
            complexity[is_front],
            (state.pop_nodes[is_front], state.pop_conns[is_front])
        )
