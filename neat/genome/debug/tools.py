from collections import defaultdict

import numpy as np


def check_array_valid(nodes, cons, input_keys, output_keys):
    nodes_dict, cons_dict = array2object(nodes, cons, input_keys, output_keys)
    # assert is_DAG(cons_dict.keys()), "The genome is not a DAG!"


def array2object(nodes, cons, input_keys, output_keys):
    """
    Convert a genome from array to dict.
    :param nodes: (N, 5)
    :param cons: (C, 4)
    :param output_keys:
    :param input_keys:
    :return: nodes_dict[key: (bias, response, act, agg)], cons_dict[(i_key, o_key): (weight, enabled)]
    """
    # update nodes_dict
    nodes_dict = {}
    for i, node in enumerate(nodes):
        if np.isnan(node[0]):
            continue
        key = int(node[0])
        assert key not in nodes_dict, f"Duplicate node key: {key}!"

        if key in input_keys:
            assert np.all(np.isnan(node[1:])), f"Input node {key} must has None bias, response, act, or agg!"
            nodes_dict[key] = (None,) * 4
        else:
            assert np.all(~np.isnan(node[1:])), f"Normal node {key} must has non-None bias, response, act, or agg!"
            bias = node[1]
            response = node[2]
            act = node[3]
            agg = node[4]
            nodes_dict[key] = (bias, response, act, agg)

    # check nodes_dict
    for i in input_keys:
        assert i in nodes_dict, f"Input node {i} not found in nodes_dict!"

    for o in output_keys:
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


def is_DAG(edges):
    all_nodes = set()
    for a, b in edges:
        if a == b:  # cycle
            return False
        all_nodes.union({a, b})

    for node in all_nodes:
        visited = {n: False for n in all_nodes}
        def dfs(n):
            if visited[n]:
                return False
            visited[n] = True
            for a, b in edges:
                if a == n:
                    if not dfs(b):
                        return False
            return True

        if not dfs(node):
            return False
    return True
