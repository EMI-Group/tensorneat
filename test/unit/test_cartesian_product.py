import numpy as np

from algorithm.hyperneat.substrate.tools import cartesian_product


def test01():
    keys1 = np.array([1, 2, 3])
    keys2 = np.array([4, 5, 6, 7])

    coors1 = np.array([
        [1, 1, 1],
        [2, 2, 2],
        [3, 3, 3]
    ])

    coors2 = np.array([
        [4, 4, 4],
        [5, 5, 5],
        [6, 6, 6],
        [7, 7, 7]
    ])

    target_coors = np.array([
        [1, 1, 1, 4, 4, 4],
        [1, 1, 1, 5, 5, 5],
        [1, 1, 1, 6, 6, 6],
        [1, 1, 1, 7, 7, 7],
        [2, 2, 2, 4, 4, 4],
        [2, 2, 2, 5, 5, 5],
        [2, 2, 2, 6, 6, 6],
        [2, 2, 2, 7, 7, 7],
        [3, 3, 3, 4, 4, 4],
        [3, 3, 3, 5, 5, 5],
        [3, 3, 3, 6, 6, 6],
        [3, 3, 3, 7, 7, 7]
    ])

    target_keys = np.array([
        [1, 4],
        [1, 5],
        [1, 6],
        [1, 7],
        [2, 4],
        [2, 5],
        [2, 6],
        [2, 7],
        [3, 4],
        [3, 5],
        [3, 6],
        [3, 7]
    ])

    new_coors, correspond_keys = cartesian_product(keys1, keys2, coors1, coors2)

    assert np.array_equal(new_coors, target_coors)
    assert np.array_equal(correspond_keys, target_keys)
