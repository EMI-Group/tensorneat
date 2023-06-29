import jax
import numpy as np


class Genome:
    def __init__(self, nodes, cons, config):
        self.config = config
        self.nodes, self.cons = array2object(nodes, cons, config)
        if config['renumber_nodes']:
            self.renumber()

    def __repr__(self):
        return f'Genome(\n' \
               f'\tinput_keys: {self.config["input_idx"]}, \n' \
               f'\toutput_keys: {self.config["output_idx"]}, \n' \
               f'\tnodes: \n\t\t' \
               f'{self.repr_nodes()} \n' \
               f'\tconnections: \n\t\t' \
               f'{self.repr_conns()} \n)'

    def repr_nodes(self):
        nodes_info = []
        for key, value in self.nodes.items():
            bias, response, act, agg = value
            act_func = self.config['activation_option_names'][int(act)] if act is not None else None
            agg_func = self.config['aggregation_option_names'][int(agg)] if agg is not None else None
            s = f"{key}: (bias: {bias}, response: {response}, act: {act_func}, agg: {agg_func})"
            nodes_info.append(s)
        return ',\n\t\t'.join(nodes_info)

    def repr_conns(self):
        conns_info = []
        for key, value in self.cons.items():
            weight, enabled = value
            s = f"{key}: (weight: {weight}, enabled: {enabled})"
            conns_info.append(s)
        return ',\n\t\t'.join(conns_info)

    def renumber(self):
        nodes2new_nodes = {}
        new_id = len(self.config['input_idx']) + len(self.config['output_idx'])
        for key in self.nodes.keys():
            if key in self.config['input_idx'] or key in self.config['output_idx']:
                nodes2new_nodes[key] = key
            else:
                nodes2new_nodes[key] = new_id
                new_id += 1

        new_nodes, new_cons = {}, {}
        for key, value in self.nodes.items():
            new_nodes[nodes2new_nodes[key]] = value
        for key, value in self.cons.items():
            i_key, o_key = key
            new_cons[(nodes2new_nodes[i_key], nodes2new_nodes[o_key])] = value
        self.nodes = new_nodes
        self.cons = new_cons


def array2object(nodes, cons, config):
    """
    Convert a genome from array to dict.
    :param nodes: (N, 5)
    :param cons: (C, 4)
    :return: nodes_dict[key: (bias, response, act, agg)], cons_dict[(i_key, o_key): (weight, enabled)]
    """
    nodes, cons = jax.device_get((nodes, cons))
    # update nodes_dict
    nodes_dict = {}
    for i, node in enumerate(nodes):
        if np.isnan(node[0]):
            continue
        key = int(node[0])
        assert key not in nodes_dict, f"Duplicate node key: {key}!"

        if key in config['input_idx']:
            assert np.all(np.isnan(node[1:])), f"Input node {key} must has None bias, response, act, or agg!"
            nodes_dict[key] = (None,) * 4
        else:
            assert np.all(
                ~np.isnan(node[1:])), f"Normal node {key} must has non-None bias, response, act, or agg!"
            bias = node[1]
            response = node[2]
            act = node[3]
            agg = node[4]
            nodes_dict[key] = (bias, response, act, agg)

    # check nodes_dict
    for i in config['input_idx']:
        assert i in nodes_dict, f"Input node {i} not found in nodes_dict!"

    for o in config['output_idx']:
        assert o in nodes_dict, f"Output node {o} not found in nodes_dict!"

    # update connections
    cons_dict = {}
    for i, con in enumerate(cons):
        if np.all(np.isnan(con)):
            pass
        elif np.all(~np.isnan(con)):
            i_key = int(con[0])
            o_key = int(con[1])
            if (i_key, o_key) in cons_dict:
                assert False, f"Duplicate connection: {(i_key, o_key)}!"
            assert i_key in nodes_dict, f"Input node {i_key} not found in nodes_dict!"
            assert o_key in nodes_dict, f"Output node {o_key} not found in nodes_dict!"
            weight = con[2]
            enabled = (con[3] == 1)
            cons_dict[(i_key, o_key)] = (weight, enabled)
        else:
            assert False, f"Connection {i} must has all None or all non-None!"

    return nodes_dict, cons_dict
