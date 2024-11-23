from typing import List, Tuple
import numpy as np

from .default import DefaultSubstrate


class MLPSubstrate(DefaultSubstrate):

    connection_type = "feedforward"

    def __init__(self, layers: List[int], coor_range: Tuple[float] = (-1, 1, -1, 1)):
        """
        layers: list of integers, the number of neurons in each layer
        coor_range: tuple of 4 floats, the range of the substrate. (x_min, x_max, y_min, y_max)
        """
        assert len(layers) >= 2, "The number of layers should be at least 2"
        for layer in layers:
            assert layer > 0, "The number of neurons in each layer should be positive"
        assert coor_range[0] < coor_range[1], "x_min should be less than x_max"
        assert coor_range[2] < coor_range[3], "y_min should be less than y_max"

        num_inputs = layers[0]
        num_outputs = layers[-1]
        query_coors, nodes, conns = analysis_substrate(layers, coor_range)
        super().__init__(num_inputs, num_outputs, query_coors, nodes, conns)


def analysis_substrate(layers, coor_range):
    x_min, x_max, y_min, y_max = coor_range
    layer_cnt = len(layers)
    y_interval = (y_max - y_min) / (layer_cnt - 1)

    # prepare nodes indices and coordinates
    node_coors = {}
    input_indices = list(range(layers[0]))
    input_coors = cal_coors(layers[0], x_min, x_max, y_min)

    output_indices = list(range(layers[0], layers[0] + layers[-1]))
    output_coors = cal_coors(layers[-1], x_min, x_max, y_max)

    if layer_cnt == 2:  # only input and output layers
        node_layers = [input_indices, output_indices]
        node_coors = [*input_coors, *output_coors]
    else:
        hidden_indices, hidden_coors = [], []
        hidden_idx = layers[0] + layers[-1]
        hidden_layers = []
        for layer_idx in range(1, layer_cnt - 1):
            y_coor = y_min + layer_idx * y_interval
            indices = list(range(hidden_idx, hidden_idx + layers[layer_idx]))
            coors = cal_coors(layers[layer_idx], x_min, x_max, y_coor)

            hidden_layers.append(indices)
            hidden_indices.extend(indices)
            hidden_coors.extend(coors)
            hidden_idx += layers[layer_idx]

        node_layers = [
            input_indices,
            *hidden_layers,
            output_indices,
        ]  # the layers of hyperneat network
        node_coors = [*input_coors, *output_coors, *hidden_coors]

    # prepare connections
    query_coors, correspond_keys = [], []
    for layer_idx in range(layer_cnt - 1):
        for i in range(layers[layer_idx]):
            for j in range(layers[layer_idx + 1]):
                neuron1 = node_layers[layer_idx][i]
                neuron2 = node_layers[layer_idx + 1][j]
                query_coors.append((*node_coors[neuron1], *node_coors[neuron2]))
                correspond_keys.append((neuron1, neuron2))

    # nodes order in TensorNEAT must be input->output->hidden
    ordered_nodes = [*node_layers[0], *node_layers[-1]]
    for layer in node_layers[1:-1]:
        ordered_nodes.extend(layer)
    nodes = np.array(ordered_nodes)[:, np.newaxis]
    conns = np.zeros(
        (len(correspond_keys), 3), dtype=np.float32
    )  # input_idx, output_idx, weight
    conns[:, :2] = correspond_keys

    query_coors = np.array(query_coors)

    return query_coors, nodes, conns


def cal_coors(neuron_cnt, x_min, x_max, y_coor):
    if neuron_cnt == 1:  # only one neuron in this layer
        return [((x_min + x_max) / 2, y_coor)]
    x_interval = (x_max - x_min) / (neuron_cnt - 1)
    return [(x_min + x_interval * i, y_coor) for i in range(neuron_cnt)]
