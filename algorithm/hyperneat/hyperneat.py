from typing import Type

import jax
import numpy as np

from .substrate import BaseSubstrate, analysis_substrate
from .hyperneat_gene import HyperNEATGene
from algorithm import State, Algorithm, neat


class HyperNEAT(Algorithm):

    def __init__(self, config, gene_type: Type[neat.BaseGene], substrate: Type[BaseSubstrate]):
        super().__init__()
        self.config = config
        self.gene_type = gene_type
        self.substrate = substrate
        self.neat = neat.NEAT(config, gene_type)

        self.tell = create_tell(self.neat)
        self.forward_transform = create_forward_transform(config, self.neat)
        self.forward = HyperNEATGene.create_forward(config)

    def setup(self, randkey, state=State()):
        state = state.update(
            below_threshold=self.config['below_threshold'],
            max_weight=self.config['max_weight']
        )

        state = self.substrate.setup(state, self.config)
        h_input_idx, h_output_idx, h_hidden_idx, query_coors, correspond_keys = analysis_substrate(state)
        h_nodes = np.concatenate((h_input_idx, h_output_idx, h_hidden_idx))[..., np.newaxis]
        h_conns = np.zeros((correspond_keys.shape[0], 3), dtype=np.float32)
        h_conns[:, 0:2] = correspond_keys

        state = state.update(
            # h is short for hyperneat
            h_input_idx=h_input_idx,
            h_output_idx=h_output_idx,
            h_hidden_idx=h_hidden_idx,
            query_coors=query_coors,
            correspond_keys=correspond_keys,
            h_nodes=h_nodes,
            h_conns=h_conns
        )
        state = self.neat.setup(randkey, state=state)

        self.config['h_input_idx'] = h_input_idx
        self.config['h_output_idx'] = h_output_idx

        return state


def create_tell(neat_instance):
    def tell(state, fitness):
        return neat_instance.tell(state, fitness)

    return tell


def create_forward_transform(config, neat_instance):
    def forward_transform(state, nodes, conns):
        t = neat_instance.forward_transform(state, nodes, conns)
        batch_forward_func = jax.vmap(neat_instance.forward, in_axes=(0, None))
        query_res = batch_forward_func(state.query_coors, t)  # hyperneat connections
        h_nodes = state.h_nodes
        h_conns = state.h_conns.at[:, 2:].set(query_res)
        return HyperNEATGene.forward_transform(state, h_nodes, h_conns)

    return forward_transform
