TensorNEAT
==========

TensorNEAT is a NEAT / HyperNEAT neuroevolution library: irregular network
topologies are *tensorized* into fixed-shape arrays, so a whole population of
differently shaped networks is evaluated at once.  Its compute layer is the
**EvoX Compiler** (``evoxc``) -- a computation graph is built once and executed
on a switchable backend (the reference ``numpy`` interpreter by default,
``iree``/``xla``/``tvm`` opt-in).  There is no JAX and no PyTorch anywhere in
the library.

Quickstart
----------

.. code-block:: python

   from tensorneat import Pipeline, algorithm, genome, problem
   from tensorneat.common import ACT

   pipeline = Pipeline(
       algorithm=algorithm.NEAT(
           pop_size=500,
           species_size=10,
           survival_threshold=0.01,
           genome=genome.DefaultGenome(
               num_inputs=2,
               num_outputs=1,
               max_nodes=12,
               output_transform=ACT.sigmoid,
           ),
       ),
       problem=problem.XOR(),
       seed=42,
       generation_limit=40,
       # fitness is maximised: this target is "error below 1e-3"
       fitness_target=-1e-3,
   )

   state = pipeline.setup()          # builds the evoxc graphs for this shape
   state, best = pipeline.auto_run(state)
   pipeline.show(state, best)

``Pipeline.setup`` creates the initial state and compiles the per-shape
executables; ``Pipeline.auto_run`` evolves until ``fitness_target`` or
``generation_limit`` is reached and returns the best genome.

API reference
-------------

.. toctree::
   :maxdepth: 2
   :caption: API reference

   source/modules

Links
-----

* `README <https://github.com/EMI-Group/tensorneat/blob/main/README.md>`_
* `Benchmark results <https://github.com/EMI-Group/tensorneat/blob/main/benchmarks/RESULTS.md>`_
* `Backends <https://github.com/EMI-Group/tensorneat/blob/main/benchmarks/BACKENDS.md>`_
