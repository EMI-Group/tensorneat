import json
import os
import warnings

from .dotdict import DotDict


class Configer:
    @classmethod
    def __load_default_config(cls):
        par_dir = os.path.dirname(os.path.abspath(__file__))
        default_config_path = os.path.join(par_dir, "./default_config.json")
        return cls.__load_config(default_config_path)

    @classmethod
    def __load_config(cls, config_path):
        with open(config_path, "r") as f:
            text = "".join(f.readlines())
        try:
            j = json.loads(text)
        except ValueError:
            raise Exception("Invalid config")
        return DotDict.from_dict(j, "root")

    @classmethod
    def __check_redundant_config(cls, default_config, config):
        for key in config:
            if key not in default_config:
                warnings.warn(f"Redundant config: {key} in {config.name}")
                continue
            if isinstance(default_config[key], DotDict):
                cls.__check_redundant_config(default_config[key], config[key])

    @classmethod
    def __complete_config(cls, default_config, config):
        for key in default_config:
            if key not in config:
                config[key] = default_config[key]
                continue
            if isinstance(default_config[key], DotDict):
                cls.__complete_config(default_config[key], config[key])

    @classmethod
    def __decorate_config(cls, config):
        if config.neat.gene.activation.options == 'all':
            config.neat.gene.activation.options = [
                "sigmoid", "tanh", "sin", "gauss", "relu", "elu", "lelu", "selu", "softplus", "identity", "clamped",
                "inv", "log", "exp", "abs", "hat", "square", "cube"
            ]
        if isinstance(config.neat.gene.activation.options, str):
            config.neat.gene.activation.options = [config.neat.gene.activation.options]

        if config.neat.gene.aggregation.options == 'all':
            config.neat.gene.aggregation.options = ["product", "sum", "max", "min", "median", "mean"]
        if isinstance(config.neat.gene.aggregation.options, str):
            config.neat.gene.aggregation.options = [config.neat.gene.aggregation.options]

    @classmethod
    def load_config(cls, config_path=None):
        default_config = cls.__load_default_config()
        if config_path is None:
            config = DotDict("root")
        elif not os.path.exists(config_path):
            warnings.warn(f"config file {config_path} not exist!")
            config = DotDict("root")
        else:
            config = cls.__load_config(config_path)

        cls.__check_redundant_config(default_config, config)
        cls.__complete_config(default_config, config)
        cls.__decorate_config(config)
        return config

    @classmethod
    def write_config(cls, config, write_path):
        text = json.dumps(config, indent=2)
        with open(write_path, "w") as f:
            f.write(text)
