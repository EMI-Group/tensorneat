"""
contains operations on a single genome. e.g. forward, mutate, crossover, etc.
"""
from .genome import create_forward, topological_sort, unflatten_connections, initialize_genomes, expand, expand_single
from .operations import create_next_generation_then_speciate
from .species import SpeciesController
