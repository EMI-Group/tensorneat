# TensorNEAT: Tensorized NEAT implementation in JAX

TensorNEAT is a powerful tool that utilizes JAX to implement the NEAT (NeuroEvolution of Augmenting Topologies) 
algorithm. It provides support for parallel execution of tasks such as network forward computation, mutation, 
and crossover at the population level.

## Requirements
* available [JAX](https://github.com/google/jax#installation) environment;
* [gymnax](https://github.com/RobertTLange/gymnax) (optional).

## Example
Simple Example for XOR problem:
```python
from config import *
from pipeline import Pipeline
from algorithm import NEAT
from algorithm.neat.gene import NormalGene, NormalGeneConfig
from problem.func_fit import XOR, FuncFitConfig

if __name__ == '__main__':
    # running config
    config = Config(
        basic=BasicConfig(
            seed=42,
            fitness_target=-1e-2,
            pop_size=10000
        ),
        neat=NeatConfig(
            inputs=2,
            outputs=1
        ),
        gene=NormalGeneConfig(),
        problem=FuncFitConfig(
            error_method='rmse'
        )
    )
    # define algorithm: NEAT with NormalGene
    algorithm = NEAT(config, NormalGene)
    # full pipeline
    pipeline = Pipeline(config, algorithm, XOR)
    # initialize state
    state = pipeline.setup()
    # run until terminate
    state, best = pipeline.auto_run(state)
    # show result
    pipeline.show(state, best)
```

Simple Example for RL envs in gymnax(CartPole-v0):
```python
import jax.numpy as jnp

from config import *
from pipeline import Pipeline
from algorithm import NEAT
from algorithm.neat.gene import NormalGene, NormalGeneConfig
from problem.rl_env import GymNaxConfig, GymNaxEnv

if __name__ == '__main__':
    conf = Config(
        basic=BasicConfig(
            seed=42,
            fitness_target=500,
            pop_size=10000
        ),
        neat=NeatConfig(
            inputs=4,
            outputs=1,
        ),
        gene=NormalGeneConfig(
            activation_default=Act.sigmoid,
            activation_options=(Act.sigmoid,),
        ),
        problem=GymNaxConfig(
            env_name='CartPole-v1',
            output_transform=lambda out: jnp.where(out[0] > 0.5, 1, 0)  # the action of cartpole is {0, 1}
        )
    )

    algorithm = NEAT(conf, NormalGene)
    pipeline = Pipeline(conf, algorithm, GymNaxEnv)
    state = pipeline.setup()
    state, best = pipeline.auto_run(state)
```

`/examples` folder contains more examples.

## TO BE COMPLETE...
