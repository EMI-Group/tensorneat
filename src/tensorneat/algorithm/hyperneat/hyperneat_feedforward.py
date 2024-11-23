"""
HyperNEAT with Feedforward Substrate and genome
"""

from typing import Callable

from .substrate import *
from .hyperneat import HyperNEAT, HyperNEATNode, HyperNEATConn
from tensorneat.common import ACT, AGG
from tensorneat.algorithm import NEAT
from tensorneat.genome import DefaultGenome


class HyperNEATFeedForward(HyperNEAT):
    def __init__(
        self,
        substrate: BaseSubstrate,
        neat: NEAT,
        weight_threshold: float = 0.3,
        max_weight: float = 5.0,
        aggregation: Callable = AGG.sum,
        activation: Callable = ACT.sigmoid,
        output_transform: Callable = ACT.sigmoid,
    ):
        assert (
            substrate.query_coors.shape[1] == neat.num_inputs
        ), "Query coors of Substrate should be equal to NEAT input size"
        
        assert substrate.connection_type == "feedforward", "Substrate should be feedforward"

        self.substrate = substrate
        self.neat = neat
        self.weight_threshold = weight_threshold
        self.max_weight = max_weight
        self.hyper_genome = DefaultGenome(
            num_inputs=substrate.num_inputs,
            num_outputs=substrate.num_outputs,
            max_nodes=substrate.nodes_cnt,
            max_conns=substrate.conns_cnt,
            node_gene=HyperNEATNode(aggregation, activation),
            conn_gene=HyperNEATConn(),
            output_transform=output_transform,
        )
        self.pop_size = neat.pop_size
