from dataclasses import dataclass
from typing import Tuple

import numpy as np

from core import Substrate, State
from config import SubstrateConfig


@dataclass(frozen=True)
class NormalSubstrateConfig(SubstrateConfig):
    input_coors: Tuple[Tuple[float]] = ((-1, -1), (0, -1), (1, -1))
    hidden_coors: Tuple[Tuple[float]] = ((-1, 0), (0, 0), (1, 0))
    output_coors: Tuple[Tuple[float]] = ((0, 1), )


class NormalSubstrate(Substrate):

    @staticmethod
    def setup(config: NormalSubstrateConfig, state: State = State()):
        return state.update(
            input_coors=np.asarray(config.input_coors, dtype=np.float32),
            output_coors=np.asarray(config.output_coors, dtype=np.float32),
            hidden_coors=np.asarray(config.hidden_coors, dtype=np.float32),
        )
