"""
aggregations, two special case need to consider:
1. extra 0s
2. full of 0s
"""
import numpy as np


def sum_agg(z):
    z = np.where(np.isnan(z), 0, z)
    return np.sum(z, axis=0)


def product_agg(z):
    z = np.where(np.isnan(z), 1, z)
    return np.prod(z, axis=0)


def max_agg(z):
    z = np.where(np.isnan(z), -np.inf, z)
    return np.max(z, axis=0)


def min_agg(z):
    z = np.where(np.isnan(z), np.inf, z)
    return np.min(z, axis=0)


def maxabs_agg(z):
    z = np.where(np.isnan(z), 0, z)
    abs_z = np.abs(z)
    max_abs_index = np.argmax(abs_z)
    return z[max_abs_index]


def median_agg(z):
    non_zero_mask = ~np.isnan(z)
    n = np.sum(non_zero_mask, axis=0)

    z = np.where(np.isnan(z), np.inf, z)
    sorted_valid_values = np.sort(z)

    if n % 2 == 0:
        return (sorted_valid_values[n // 2 - 1] + sorted_valid_values[n // 2]) / 2
    else:
        return sorted_valid_values[n // 2]


def mean_agg(z):
    non_zero_mask = ~np.isnan(z)
    valid_values_sum = sum_agg(z)
    valid_values_count = np.sum(non_zero_mask, axis=0)
    mean_without_zeros = valid_values_sum / valid_values_count
    return mean_without_zeros


AGG_TOTAL_LIST = [sum_agg, product_agg, max_agg, min_agg, maxabs_agg, median_agg, mean_agg]

agg_name2key = {
    'sum': 0,
    'product': 1,
    'max': 2,
    'min': 3,
    'maxabs': 4,
    'median': 5,
    'mean': 6,
}


def agg(idx, z):
    idx = np.asarray(idx, dtype=np.int32)

    if np.all(z == 0.):
        return 0
    else:
        return AGG_TOTAL_LIST[idx](z)


if __name__ == '__main__':
    array = np.asarray([1, 2, np.nan, np.nan, 3, 4, 5, np.nan, np.nan, np.nan, np.nan], dtype=np.float32)
    for names in agg_name2key.keys():
        print(names, agg(agg_name2key[names], array))

    array2 = np.asarray([0, 0, 0, 0], dtype=np.float32)
    for names in agg_name2key.keys():
        print(names, agg(agg_name2key[names], array2))
