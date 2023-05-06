from functools import partial
from typing import Tuple

import numpy as np
from numpy.typing import NDArray

I_INT = np.iinfo(np.int32).max  # infinite int


def flatten_connections(keys, connections):
    """
    flatten the (2, N, N) connections to (N * N, 4)
    :param keys:
    :param connections:
    :return:
    the first two columns are the index of the node
    the 3rd column is the weight, and the 4th column is the enabled status
    """
    indices_x, indices_y = np.meshgrid(keys, keys, indexing='ij')
    indices = np.stack((indices_x, indices_y), axis=-1).reshape(-1, 2)

    # make (2, N, N) to (N, N, 2)
    con = np.transpose(connections, (1, 2, 0))
    # make (N, N, 2) to (N * N, 2)
    con = np.reshape(con, (-1, 2))

    con = np.concatenate((indices, con), axis=1)
    return con


def unflatten_connections(N, cons):
    """
    restore the (N * N, 4) connections to (2, N, N)
    :param N:
    :param cons:
    :return:
    """
    cons = cons[:, 2:]  # remove the indices
    unflatten_cons = np.moveaxis(cons.reshape(N, N, 2), -1, 0)
    return unflatten_cons


def set_operation_analysis(ar1: NDArray, ar2: NDArray) -> Tuple[NDArray, NDArray, NDArray]:
    """
    Analyze the intersection and union of two arrays by returning their sorted concatenation indices,
    intersection mask, and union mask.

    :param ar1: JAX array of shape (N, M)
        First input array. Should have the same shape as ar2.
    :param ar2: JAX array of shape (N, M)
        Second input array. Should have the same shape as ar1.
    :return: tuple of 3 arrays
        - sorted_indices: Indices that would sort the concatenation of ar1 and ar2.
        - intersect_mask: A boolean array indicating the positions of the common elements between ar1 and ar2
                          in the sorted concatenation.
        - union_mask: A boolean array indicating the positions of the unique elements in the union of ar1 and ar2
                      in the sorted concatenation.

    Examples:
        a = np.array([[1, 2], [3, 4], [5, 6]])
        b = np.array([[1, 2], [7, 8], [9, 10]])

        sorted_indices, intersect_mask, union_mask = set_operation_analysis(a, b)

        sorted_indices -> array([0, 1, 2, 3, 4, 5])
        intersect_mask -> array([True, False, False, False, False, False])
        union_mask -> array([False, True, True, True, True, True])
    """
    ar = np.concatenate((ar1, ar2), axis=0)
    sorted_indices = np.lexsort(ar.T[::-1])
    aux = ar[sorted_indices]
    aux = np.concatenate((aux, np.full((1, ar1.shape[1]), np.nan)), axis=0)
    nan_mask = np.any(np.isnan(aux), axis=1)

    fr, sr = aux[:-1], aux[1:]  # first row, second row
    intersect_mask = np.all(fr == sr, axis=1) & ~nan_mask[:-1]
    union_mask = np.any(fr != sr, axis=1) & ~nan_mask[:-1]
    return sorted_indices, intersect_mask, union_mask


def fetch_first(mask, default=I_INT) -> NDArray:
    """
    fetch the first True index
    :param mask: array of bool
    :param default: the default value if no element satisfying the condition
    :return: the index of the first element satisfying the condition. if no element satisfying the condition, return I_INT
    example:
    >>> a = np.array([1, 2, 3, 4, 5])
    >>> fetch_first(a > 3)
    3
    >>> fetch_first(a > 30)
    I_INT
    """
    idx = np.argmax(mask)
    return np.where(mask[idx], idx, default)


def fetch_last(mask, default=I_INT) -> NDArray:
    """
    similar to fetch_first, but fetch the last True index
    """
    reversed_idx = fetch_first(mask[::-1], default)
    return np.where(reversed_idx == default, default, mask.shape[0] - reversed_idx - 1)


def fetch_random(mask, default=I_INT) -> NDArray:
    """
    similar to fetch_first, but fetch a random True index
    """
    true_cnt = np.sum(mask)
    if true_cnt == 0:
        return default
    cumsum = np.cumsum(mask)
    target = np.random.randint(1, true_cnt + 1, size=())
    return fetch_first(cumsum >= target, default)


if __name__ == '__main__':
    a = np.array([1, 2, 3, 4, 5])
    print(fetch_first(a > 3))
    print(fetch_first(a > 30))

    print(fetch_last(a > 3))
    print(fetch_last(a > 30))

    for t in [-1, 0, 1, 2, 3, 4, 5]:
        for _ in range(10):
            print(t, fetch_random(a > t))
