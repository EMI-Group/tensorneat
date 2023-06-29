"""
contains operations on a single genome. e.g. forward, mutate, crossover, etc.
"""
from .genome import create_forward_function, topological_sort, unflatten_connections
from .population import update_species, create_next_generation, speciate
