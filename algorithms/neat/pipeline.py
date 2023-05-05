import jax

from .species import SpeciesController
from .genome import create_initialize_function, create_mutate_function, create_forward_function


class Pipeline:
    """
    Neat algorithm pipeline.
    """

    def __init__(self, config):
        self.config = config
        self.N = config.basic.init_maximum_nodes

        self.species_controller = SpeciesController(config)
        self.initialize_func = create_initialize_function(config)
        self.pop_nodes, self.pop_connections, self.input_idx, self.output_idx = self.initialize_func()
        self.mutate_func = create_mutate_function(config, self.input_idx, self.output_idx, batch=True)

        self.generation = 0

        self.species_controller.speciate(self.pop_nodes, self.pop_connections, self.generation)

    def ask(self, batch: bool):
        """
        Create a forward function for the population.
        :param batch:
        :return:
        Algorithm gives the population a forward function, then environment gives back the fitnesses.
        """
        func = create_forward_function(self.pop_nodes, self.pop_connections, self.N, self.input_idx, self.output_idx,
                                       batch=batch)
        return func

    def tell(self, fitnesses):
        self.generation += 1
        print(type(fitnesses), fitnesses)
        self.species_controller.update_species_fitnesses(fitnesses)


