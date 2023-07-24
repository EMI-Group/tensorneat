from dataclasses import dataclass


@dataclass(frozen=True)
class BasicConfig:
    seed: int = 42
    fitness_target: float = 1
    generation_limit: int = 1000
    pop_size: int = 100

    def __post_init__(self):
        assert self.pop_size > 0, "the population size must be greater than 0"


@dataclass(frozen=True)
class NeatConfig:
    network_type: str = "feedforward"
    inputs: int = 2
    outputs: int = 1
    maximum_nodes: int = 50
    maximum_conns: int = 100
    maximum_species: int = 10

    # genome config
    compatibility_disjoint: float = 1
    compatibility_weight: float = 0.5
    conn_add: float = 0.4
    conn_delete: float = 0
    node_add: float = 0.2
    node_delete: float = 0

    # species config
    compatibility_threshold: float = 3.5
    species_elitism: int = 2
    max_stagnation: int = 15
    genome_elitism: int = 2
    survival_threshold: float = 0.2
    min_species_size: int = 1
    spawn_number_change_rate: float = 0.5

    def __post_init__(self):
        assert self.network_type in ["feedforward", "recurrent"], "the network type must be feedforward or recurrent"

        assert self.inputs > 0, "the inputs number of neat must be greater than 0"
        assert self.outputs > 0, "the outputs number of neat must be greater than 0"

        assert self.maximum_nodes > 0, "the maximum nodes must be greater than 0"
        assert self.maximum_conns > 0, "the maximum connections must be greater than 0"
        assert self.maximum_species > 0, "the maximum species must be greater than 0"

        assert self.compatibility_disjoint > 0, "the compatibility disjoint must be greater than 0"
        assert self.compatibility_weight > 0, "the compatibility weight must be greater than 0"
        assert self.conn_add >= 0, "the connection add probability must be greater than 0"
        assert self.conn_delete >= 0, "the connection delete probability must be greater than 0"
        assert self.node_add >= 0, "the node add probability must be greater than 0"
        assert self.node_delete >= 0, "the node delete probability must be greater than 0"

        assert self.compatibility_threshold > 0, "the compatibility threshold must be greater than 0"
        assert self.species_elitism > 0, "the species elitism must be greater than 0"
        assert self.max_stagnation > 0, "the max stagnation must be greater than 0"
        assert self.genome_elitism > 0, "the genome elitism must be greater than 0"
        assert self.survival_threshold > 0, "the survival threshold must be greater than 0"
        assert self.min_species_size > 0, "the min species size must be greater than 0"
        assert self.spawn_number_change_rate > 0, "the spawn number change rate must be greater than 0"


@dataclass(frozen=True)
class HyperNeatConfig:
    below_threshold: float = 0.2
    max_weight: float = 3
    activation: str = "sigmoid"
    aggregation: str = "sum"
    activate_times: int = 5
    inputs: int = 2
    outputs: int = 1

    def __post_init__(self):
        assert self.below_threshold > 0, "the below threshold must be greater than 0"
        assert self.max_weight > 0, "the max weight must be greater than 0"
        assert self.activate_times > 0, "the activate times must be greater than 0"
        assert self.inputs > 0, "the inputs number of hyper neat must be greater than 0"
        assert self.outputs > 0, "the outputs number of hyper neat must be greater than 0"


@dataclass(frozen=True)
class GeneConfig:
    pass

@dataclass(frozen=True)
class SubstrateConfig:
    pass


@dataclass(frozen=True)
class Config:
    basic: BasicConfig = BasicConfig()
    neat: NeatConfig = NeatConfig()
    hyper_neat: HyperNeatConfig = HyperNeatConfig()
    gene: GeneConfig = GeneConfig()
    substrate: SubstrateConfig = SubstrateConfig()
