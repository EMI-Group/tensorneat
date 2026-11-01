import numpy as np
from .default import DefaultSubstrate


class FullSubstrate(DefaultSubstrate):

    connection_type = "recurrent"

    def __init__(
        self,
        input_coors=((-1, -1), (0, -1), (1, -1)),
        hidden_coors=((-1, 0), (0, 0), (1, 0)),
        output_coors=((0, 1),),
    ):
        query_coors, nodes, conns = analysis_substrate(
            input_coors, output_coors, hidden_coors
        )
        super().__init__(len(input_coors), len(output_coors), query_coors, nodes, conns)


def analysis_substrate(input_coors, output_coors, hidden_coors):
    input_coors = np.array(input_coors)
    output_coors = np.array(output_coors)
    hidden_coors = np.array(hidden_coors)

    cd = input_coors.shape[1]  # coordinate dimensions
    si = input_coors.shape[0]  # input coordinate size
    so = output_coors.shape[0]  # output coordinate size
    sh = hidden_coors.shape[0]  # hidden coordinate size

    input_idx = np.arange(si)
    output_idx = np.arange(si, si + so)
    hidden_idx = np.arange(si + so, si + so + sh)

    total_conns = si * sh + sh * sh + sh * so
    query_coors = np.zeros((total_conns, cd * 2))
    correspond_keys = np.zeros((total_conns, 2))

    # connect input to hidden
    aux_coors, aux_keys = cartesian_product(
        input_idx, hidden_idx, input_coors, hidden_coors
    )
    query_coors[0 : si * sh, :] = aux_coors
    correspond_keys[0 : si * sh, :] = aux_keys

    # connect hidden to hidden
    aux_coors, aux_keys = cartesian_product(
        hidden_idx, hidden_idx, hidden_coors, hidden_coors
    )
    query_coors[si * sh : si * sh + sh * sh, :] = aux_coors
    correspond_keys[si * sh : si * sh + sh * sh, :] = aux_keys

    # connect hidden to output
    aux_coors, aux_keys = cartesian_product(
        hidden_idx, output_idx, hidden_coors, output_coors
    )
    query_coors[si * sh + sh * sh :, :] = aux_coors
    correspond_keys[si * sh + sh * sh :, :] = aux_keys

    nodes = np.concatenate((input_idx, output_idx, hidden_idx))[..., np.newaxis]
    conns = np.zeros(
        (correspond_keys.shape[0], 3), dtype=np.float32
    )  # input_idx, output_idx, weight
    conns[:, :2] = correspond_keys

    # print(query_coors, nodes, conns)
    return query_coors, nodes, conns


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
