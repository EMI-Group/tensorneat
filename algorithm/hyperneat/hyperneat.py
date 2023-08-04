from typing import Type

import jax
from jax import numpy as jnp, Array, vmap
import numpy as np

from config import Config, HyperNeatConfig
from core import Algorithm, Substrate, State, Genome, Gene
from utils import Activation, Aggregation
from .substrate import analysis_substrate
from algorithm import NEAT


class HyperNEAT(Algorithm):

    def __init__(self, config: Config, gene: Type[Gene], substrate: Type[Substrate]):
        self.config = config
        self.neat = NEAT(config, gene)
        self.substrate = substrate

    def setup(self, randkey, state=State()):
        neat_key, randkey = jax.random.split(randkey)
        state = state.update(
            below_threshold=self.config.hyperneat.below_threshold,
            max_weight=self.config.hyperneat.max_weight,
        )
        state = self.neat.setup(neat_key, state)
        state = self.substrate.setup(self.config.substrate, state)

        assert self.config.hyperneat.inputs + 1 == state.input_coors.shape[0]  # +1 for bias
        assert self.config.hyperneat.outputs == state.output_coors.shape[0]

        h_input_idx, h_output_idx, h_hidden_idx, query_coors, correspond_keys = analysis_substrate(state)
        h_nodes = np.concatenate((h_input_idx, h_output_idx, h_hidden_idx))[..., np.newaxis]
        h_conns = np.zeros((correspond_keys.shape[0], 3), dtype=np.float32)
        h_conns[:, 0:2] = correspond_keys

        state = state.update(
            h_input_idx=h_input_idx,
            h_output_idx=h_output_idx,
            h_hidden_idx=h_hidden_idx,
            h_nodes=h_nodes,
            h_conns=h_conns,
            query_coors=query_coors,
        )

        return state

    def ask_algorithm(self, state: State):
        return state.pop_genomes

    def tell_algorithm(self, state: State, fitness):
        return self.neat.tell(state, fitness)

    def forward(self, state, inputs: Array, transformed: Array):
        return HyperNEATGene.forward(self.config.hyperneat, state, inputs, transformed)

    def forward_transform(self, state: State, genome: Genome):
        t = self.neat.forward_transform(state, genome)
        query_res = vmap(self.neat.forward, in_axes=(None, 0, None))(state, state.query_coors, t)

        # mute the connection with weight below threshold
        query_res = jnp.where((-state.below_threshold < query_res) & (query_res < state.below_threshold), 0., query_res)

        # make query res in range [-max_weight, max_weight]
        query_res = jnp.where(query_res > 0, query_res - state.below_threshold, query_res)
        query_res = jnp.where(query_res < 0, query_res + state.below_threshold, query_res)
        query_res = query_res / (1 - state.below_threshold) * state.max_weight

        h_conns = state.h_conns.at[:, 2:].set(query_res)

        return HyperNEATGene.forward_transform(Genome(state.h_nodes, h_conns))


class HyperNEATGene:
    node_attrs = []  # no node attributes
    conn_attrs = ['weight']

    @staticmethod
    def forward_transform(genome: Genome):
        N = genome.nodes.shape[0]
        u_conns = jnp.zeros((N, N), dtype=jnp.float32)

        in_keys = jnp.asarray(genome.conns[:, 0], jnp.int32)
        out_keys = jnp.asarray(genome.conns[:, 1], jnp.int32)
        weights = genome.conns[:, 2]

        u_conns = u_conns.at[in_keys, out_keys].set(weights)
        return genome.nodes, u_conns

    @staticmethod
    def forward(config: HyperNeatConfig, state: State, inputs, transformed):
        act = Activation.name2func[config.activation]
        agg = Aggregation.name2func[config.aggregation]

        batch_act, batch_agg = jax.vmap(act), jax.vmap(agg)

        nodes, weights = transformed

        inputs_with_bias = jnp.concatenate((inputs, jnp.ones((1,))), axis=0)

        input_idx = state.h_input_idx
        output_idx = state.h_output_idx

        N = nodes.shape[0]
        vals = jnp.full((N,), 0.)

        def body_func(i, values):
            values = values.at[input_idx].set(inputs_with_bias)
            nodes_ins = values * weights.T
            values = batch_agg(nodes_ins)  # z = agg(ins)
            values = values * nodes[:, 2] + nodes[:, 1]  # z = z * response + bias
            values = batch_act(values)  # z = act(z)
            return values

        vals = jax.lax.fori_loop(0, config.activate_times, body_func, vals)
        return vals[output_idx]
