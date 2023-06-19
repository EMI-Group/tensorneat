import jax

from configs.configer import Configer
from .genome.genome_ import initialize_genomes


class Pipeline:
    """
    Neat algorithm pipeline.
    """

    def __init__(self, config, seed=42):
        self.randkey = jax.random.PRNGKey(seed)

        self.config = config  # global config
        self.jit_config = Configer.create_jit_config(config)  # config used in jit-able functions
        self.N = self.config["init_maximum_nodes"]
        self.C = self.config["init_maximum_connections"]
        self.S = self.config["init_maximum_species"]

        self.generation = 0
        self.best_genome = None

        self.pop_nodes, self.pop_cons = initialize_genomes(self.N, self.C, self.config)

        print(self.pop_nodes, self.pop_cons, sep='\n')
        print(self.jit_config)
