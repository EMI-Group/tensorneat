import os
import warnings
import configparser

import numpy as np

from .activations import refactor_act
from .aggregations import refactor_agg

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
    "bias_init_stdev",
    "bias_mutate_power",
    "bias_mutate_rate",
    "bias_replace_rate",
    "response_init_mean",
    "response_init_stdev",
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
    "weight_init_stdev",
    "weight_mutate_power",
    "weight_mutate_rate",
    "weight_replace_rate",
    "enable_mutate_rate",
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

        refactor_act(config)
        refactor_agg(config)
        input_idx = np.arange(config['num_inputs'])
        output_idx = np.arange(config['num_inputs'], config['num_inputs'] + config['num_outputs'])
        config['input_idx'] = input_idx
        config['output_idx'] = output_idx
        return config

    @classmethod
    def create_jit_config(cls, config):
        jit_config = {k: config[k] for k in jit_config_keys}

        return jit_config
