from neat.genome.aggregations import *

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


def refactor_agg(config):
    config['aggregation_default'] = agg_name2key[config['aggregation_default']]
    config['aggregation_options'] = [
        agg_name2key[act_name] for act_name in config['aggregation_options']
    ]
