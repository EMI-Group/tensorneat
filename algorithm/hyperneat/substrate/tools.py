from typing import Type

import numpy as np

from .base import BaseSubstrate


def analysis_substrate(state):
    cd = state.input_coors.shape[1]  # coordinate dimensions
    si = state.input_coors.shape[0]  # input coordinate size
    so = state.output_coors.shape[0]  # output coordinate size
    sh = state.hidden_coors.shape[0]  # hidden coordinate size

    input_idx = np.arange(si)
    output_idx = np.arange(si, si + so)
    hidden_idx = np.arange(si + so, si + so + sh)

    total_conns = si * sh + sh * sh + sh * so
    query_coors = np.zeros((total_conns, cd * 2))
    correspond_keys = np.zeros((total_conns, 2))

    # connect input to hidden
    aux_coors, aux_keys = cartesian_product(input_idx, hidden_idx, state.input_coors, state.hidden_coors)
    query_coors[0: si * sh, :] = aux_coors
    correspond_keys[0: si * sh, :] = aux_keys

    # connect hidden to hidden
    aux_coors, aux_keys = cartesian_product(hidden_idx, hidden_idx, state.hidden_coors, state.hidden_coors)
    query_coors[si * sh: si * sh + sh * sh, :] = aux_coors
    correspond_keys[si * sh: si * sh + sh * sh, :] = aux_keys

    # connect hidden to output
    aux_coors, aux_keys = cartesian_product(hidden_idx, output_idx, state.hidden_coors, state.output_coors)
    query_coors[si * sh + sh * sh:, :] = aux_coors
    correspond_keys[si * sh + sh * sh:, :] = aux_keys

    return input_idx, output_idx, hidden_idx, query_coors, correspond_keys


def cartesian_product(keys1, keys2, coors1, coors2):
    len1 = keys1.shape[0]
    len2 = keys2.shape[0]

    repeated_coors1 = np.repeat(coors1, len2, axis=0)
    repeated_keys1 = np.repeat(keys1, len2)

    tiled_coors2 = np.tile(coors2, (len1, 1))
    tiled_keys2 = np.tile(keys2, len1)

    new_coors = np.concatenate((repeated_coors1, tiled_coors2), axis=1)
    correspond_keys = np.column_stack((repeated_keys1, tiled_keys2))

    return new_coors, correspond_keys
