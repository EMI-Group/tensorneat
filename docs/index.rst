TensorNEAT
==========

TensorNEAT is a NEAT / HyperNEAT neuroevolution library: irregular network
topologies are *tensorized* into fixed-shape arrays, so a whole population of
differently shaped networks is evaluated at once.  Its compute layer is the
**EvoX Compiler** (``evoxc``) -- a computation graph is built once per shape and
then executed once per generation on a switchable backend and device.
``backend="numpy"`` (the default) is the reference interpreter and needs no
optional dependency; ``backend="iree"`` is the compiled backend that works end
to end here (``uv sync --extra iree``) and, on the CPU, is still slower per
generation than the numpy interpreter (the CPU-side record, on IREE 3.13, is
``benchmarks/BACKENDS.md``); ``backend="xla"`` requires a user-supplied PJRT C
API plugin ``.so``, named either by the ``EVOXC_PJRT_PLUGIN`` environment
variable or by the
``Pipeline(..., compile_options={"plugin_path": "/path/to/plugin.so"})``
(the compile option wins over the variable), and none ships with the
package; ``tvm`` is untested here.  There is no JAX and no PyTorch
anywhere in the library.

``device=`` is the second axis.  A device kind (``"cpu"``, ``"cuda"``,
``"rocm"``) or an ``evoxc.Device`` is forwarded to ``evoxc.build`` and the
runtime tensors are placed there once, after ``Pipeline.setup``, with nothing
ever falling back to CPU.  The kind also chooses the kernel formulations baked
into the generation graph: a GPU-class device resolves to the accelerator
spellings -- the index-free distance formulation over a grouped centre batch
(``genome.operations.distance.index_free`` and
``genome.operations.distance.grouped``),
scatter-free schedule writes (``common.schedule_accelerator``, whose
``kernel_mode`` ``tensorneat.common`` re-exports) and a scatter-free cycle
predicate (``genome.operations.mutation.cycle_accelerator``) -- while every
other kind keeps the host spellings.  A compiled device path needs that same
``uv sync --extra iree`` environment.  The measured accelerator record -- a ROCm
``gfx942`` device is faster per generation than the numpy interpreter on all six
shipped cases, against a one-off ``iree-compile`` per static shape -- is
``benchmarks/GPU.md``.

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

``host_payload`` decides what ``Pipeline.step`` hands back as the evaluated
population.  The default (``host_payload=True``) copies the four gene arrays to
host numpy, so ``previous_pop`` and everything built on it -- ``best_genome``,
the checkpoints, the report -- is plain numpy; ``Pipeline(host_payload=False)``,
or a per-call ``step(..., host_payload=False)``, skips that copy and returns the
payload as the backend left it -- numpy on the reference interpreter,
device-resident tensors on a device backend -- and the reporting paths
materialise it only on the generations that need it (a new best, a checkpoint,
the problem's ``show_details`` hook), so materialise a field with
``common.tools.to_numpy`` when you want a host array yourself.  The fitness
vector is always host.

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
* `Accelerator (GPU) results <https://github.com/EMI-Group/tensorneat/blob/main/benchmarks/GPU.md>`_
