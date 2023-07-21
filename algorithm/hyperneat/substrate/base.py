import numpy as np


class BaseSubstrate:

    @staticmethod
    def setup(state, config):
        return state.update(
            input_coors=np.asarray(config['input_coors'], dtype=np.float32),
            output_coors=np.asarray(config['output_coors'], dtype=np.float32),
            hidden_coors=np.asarray(config['hidden_coors'], dtype=np.float32),
        )
