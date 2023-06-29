"""
contains operations on a single genome. e.g. forward, mutate, crossover, etc.
"""
from .genome import create_forward_function, topological_sort, unflatten_connections, initialize_genomes
from .population import update_species, create_next_generation, speciate

from .genome.activations import act_name2func
from .genome.aggregations import agg_name2func

from .visualize import Genome
