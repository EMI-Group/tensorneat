import numpy as np

I_INT = np.iinfo(np.int32).max  # infinite int


def flatten_connections(keys, connections):
    indices_x, indices_y = np.meshgrid(keys, keys, indexing='ij')
    indices = np.stack((indices_x, indices_y), axis=-1).reshape(-1, 2)

    # make (2, N, N) to (N, N, 2)
    con = np.transpose(connections, (1, 2, 0))
    # make (N, N, 2) to (N * N, 2)
    con = np.reshape(con, (-1, 2))

    con = np.concatenate((indices, con), axis=1)
    return con


def unflatten_connections(N, cons):
    cons = cons[:, 2:]  # remove the indices
    unflatten_cons = np.moveaxis(cons.reshape(N, N, 2), -1, 0)
    return unflatten_cons


def set_operation_analysis(ar1, ar2):
    ar = np.concatenate((ar1, ar2), axis=0)
    sorted_indices = np.lexsort(ar.T[::-1])
    aux = ar[sorted_indices]
    aux = np.concatenate((aux, np.full((1, ar1.shape[1]), np.nan)), axis=0)
    nan_mask = np.any(np.isnan(aux), axis=1)

    fr, sr = aux[:-1], aux[1:]  # first row, second row
    intersect_mask = np.all(fr == sr, axis=1) & ~nan_mask[:-1]
    union_mask = np.any(fr != sr, axis=1) & ~nan_mask[:-1]
    return sorted_indices, intersect_mask, union_mask


def fetch_first(mask, default=I_INT):
    idx = np.argmax(mask)
    return np.where(mask[idx], idx, default)


def fetch_last(mask, default=I_INT):
    reversed_idx = fetch_first(mask[::-1], default)
    return np.where(reversed_idx == -1, -1, mask.shape[0] - reversed_idx - 1)


def fetch_random(rand_key, mask, default=I_INT):
    """
    similar to fetch_first, but fetch a random True index
    """
    true_cnt = np.sum(mask)
    cumsum = np.cumsum(mask)
    target = np.random.randint(rand_key, shape=(), minval=0, maxval=true_cnt + 1)
    return fetch_first(cumsum >= target, default)
