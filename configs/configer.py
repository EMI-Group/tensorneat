import os
import warnings
import configparser

import numpy as np

from algorithms.neat.genome.activations import act_name2func
from algorithms.neat.genome.aggregations import agg_name2func

# Configuration used in jit-able functions. The change of values will not cause the re-compilation of JAX.
jit_config_keys = [
    "input_idx",
    "output_idx",
    "compatibility_disjoint",
    "compatibility_weight",
    "conn_add_prob",
    "conn_add_trials",
    "conn_delete_prob",
    "node_add_prob",
    "node_delete_prob",
    "compatibility_threshold",
    "bias_init_mean",
    "bias_init_std",
    "bias_mutate_power",
    "bias_mutate_rate",
    "bias_replace_rate",
    "response_init_mean",
    "response_init_std",
    "response_mutate_power",
    "response_mutate_rate",
    "response_replace_rate",
    "activation_default",
    "activation_options",
    "activation_replace_rate",
    "aggregation_default",
    "aggregation_options",
    "aggregation_replace_rate",
    "weight_init_mean",
    "weight_init_std",
    "weight_mutate_power",
    "weight_mutate_rate",
    "weight_replace_rate",
    "enable_mutate_rate",
    "max_stagnation",
    "pop_size",
    "genome_elitism",
    "survival_threshold",
    "species_elitism"
]


class Configer:

    @classmethod
    def __load_default_config(cls):
        par_dir = os.path.dirname(os.path.abspath(__file__))
        default_config_path = os.path.join(par_dir, "default_config.ini")
        return cls.__load_config(default_config_path)

    @classmethod
    def __load_config(cls, config_path):
        c = configparser.ConfigParser()
        c.read(config_path)
        config = {}

        for section in c.sections():
            for key, value in c.items(section):
                config[key] = eval(value)

        return config

    @classmethod
    def __check_redundant_config(cls, default_config, config):
        for key in config:
            if key not in default_config:
                warnings.warn(f"Redundant config: {key} in {config.name}")

    @classmethod
    def __complete_config(cls, default_config, config):
        for key in default_config:
            if key not in config:
                config[key] = default_config[key]

    @classmethod
    def load_config(cls, config_path=None):
        default_config = cls.__load_default_config()
        if config_path is None:
            config = {}
        elif not os.path.exists(config_path):
            warnings.warn(f"config file {config_path} not exist!")
            config = {}
        else:
            config = cls.__load_config(config_path)

        cls.__check_redundant_config(default_config, config)
        cls.__complete_config(default_config, config)

        cls.refactor_activation(config)
        cls.refactor_aggregation(config)

        config['input_idx'] = np.arange(config['num_inputs'])
        config['output_idx'] = np.arange(config['num_inputs'], config['num_inputs'] + config['num_outputs'])

        return config

    @classmethod
    def refactor_activation(cls, config):
        config['activation_default'] = 0
        config['activation_options'] = np.arange(len(config['activation_option_names']))
        config['activation_funcs'] = [act_name2func[name] for name in config['activation_option_names']]

    @classmethod
    def refactor_aggregation(cls, config):
        config['aggregation_default'] = 0
        config['aggregation_options'] = np.arange(len(config['aggregation_option_names']))
        config['aggregation_funcs'] = [agg_name2func[name] for name in config['aggregation_option_names']]

    @classmethod
    def create_jit_config(cls, config):
        jit_config = {k: config[k] for k in jit_config_keys}

        return jit_config
