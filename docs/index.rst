TensorNEAT
==========

TensorNEAT is a NEAT / HyperNEAT neuroevolution library: irregular network
topologies are *tensorized* into fixed-shape arrays, so a whole population of
differently shaped networks is evaluated at once.  Its compute layer is the
**EvoX Compiler** (``evoxc``) -- a computation graph is built once per shape and
then executed once per generation on a switchable backend.  ``backend="numpy"``
(the default) is the reference interpreter; ``backend="iree"`` is the compiled
backend that works end to end here (``uv sync --extra iree``); ``backend="xla"``
requires a user-supplied PJRT C API plugin ``.so``, named either by the
``EVOXC_PJRT_PLUGIN`` environment variable or by the
``Pipeline(..., compile_options={"plugin_path": "/path/to/plugin.so"})``
(the compile option wins over the variable), and none ships with the
package; ``tvm`` is untested here.  There is no JAX and no PyTorch
anywhere in the library.

Quickstart
----------

.. code-block:: python

   from tensorneat import Pipeline, algorithm, genome, problem
   from tensorneat.common import ACT

   pipeline = Pipeline(
       algorithm=algorithm.NEAT(
           pop_size=200,
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
       generation_limit=200,
       # fitness is maximised: this target is "error below 1e-3"
       fitness_target=-1e-3,
   )

   state = pipeline.setup()          # host-side state only -- no graph is built yet
   state, best = pipeline.auto_run(state)
   pipeline.show(state, best)

``Pipeline.setup`` creates the host-side state and builds nothing; the evoxc
executable for a shape signature is compiled lazily by the first ``Pipeline.step``
(or the start of ``Pipeline.auto_run``) and then reused for every generation of
that shape.  ``Pipeline.auto_run`` evolves until
``fitness_target`` or ``generation_limit`` is reached and returns the best genome
as a one-individual, batch-first payload.

Exact gradients through an evolved topology are host-side as well:
``genome.DefaultGenome`` and ``genome.RecurrentGenome`` inherit
``BaseGenomeHost.grad_from_payload`` (``tensorneat.genome.host``), which derives
the schedule on the host and returns ``(node_attr_grads, conn_attr_grads)`` for
one batch-first payload.

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
