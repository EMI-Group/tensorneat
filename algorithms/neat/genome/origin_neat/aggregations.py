"""
Has the built-in aggregation functions, code for using them,
and code for adding new user-defined ones.
"""

def sum_aggregation(x):
    return sum(x)


aggregation_dict = {
    'sum': sum_aggregation,
}

full_aggregation_list = list(aggregation_dict.keys())