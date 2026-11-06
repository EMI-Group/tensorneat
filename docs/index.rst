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
API plugin ``.so``, named by the ``EVOXC_PJRT_PLUGIN`` environment variable or by
``compile_options={"plugin_path": "/path/to/plugin.so"}``, and none ships with
the package; ``tvm`` is untested here.  There is no JAX and no PyTorch
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
other kind keeps the host spellings and its per-generation cost.  Each compiled
path needs its own install: see `Running on an accelerator`_ for the exact
commands.  The measured accelerator record -- a ROCm ``gfx942`` device is faster
per generation than the numpy interpreter on all six shipped cases, on both
compiled backends -- is ``benchmarks/GPU.md``.

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

Running on an accelerator
-------------------------

``device=`` is the only extra argument: ``Pipeline.setup`` still builds and runs
its host-side graphs on CPU (the device-selecting ``device``/``target_backends``
keys are stripped for that phase), the runtime tensors are placed on the device
once afterwards, and nothing transfers implicitly -- only the reported values
come back to host, plus the four gene arrays when ``host_payload=True``.  The
generation graph is compiled once per static shape, and that one-off compile is
excluded from every per-generation figure quoted below and in the benchmark
records.

IREE on a ROCm device -- ``target_backends`` and ``device`` must agree, and
nothing falls back if they do not:

.. code-block:: bash

   uv sync --extra iree

.. code-block:: python

   pipeline = Pipeline(
       algorithm=algorithm.NEAT(...),
       problem=problem.XOR(),
       backend="iree",
       device="rocm",
       compile_options={"target_backends": ["rocm"]},
   )

XLA on the same device -- the ``xla`` backend is driven by a PJRT C API plugin
``.so`` that no package of this project ships, so the plugin is installed into
the project environment after the sync, because a later ``uv sync`` prunes a
package it does not track:

.. code-block:: bash

   uv sync --extra iree                      # or your own extras -- this creates the project venv
   uv pip install --no-deps jax-rocm7-pjrt   # ships jax_plugins/xla_rocm7/xla_rocm_plugin.so
   export EVOXC_PJRT_PLUGIN=$(uv run python -c "import sysconfig; print(sysconfig.get_paths()['purelib'])")/jax_plugins/xla_rocm7/xla_rocm_plugin.so

.. code-block:: python

   pipeline = Pipeline(..., backend="xla", device="rocm")

On an NVIDIA box the plugin package is ``jax-cuda12-pjrt`` and the device kind
is ``"cuda"``.  The pinned ``evoxc`` (the tag ``uv`` resolves) does not scan
``site-packages`` for plugins: it resolves the plugin from
``compile_options["plugin_path"]``, then from ``EVOXC_PJRT_PLUGIN``, then from a
few well-known library directories -- and its availability probe takes no
options and reads only that variable, so export it as above rather than passing
``plugin_path`` alone.  A newer ``evoxc`` discovers an installed
``jax_plugins/*/*_plugin.so`` by itself.

Both compiled paths are bit-identical to the reference interpreter at generation
1 (integer keys exact, attributes within two float32 ulps).  On this box's AMD
Instinct MI300X a generation costs 3.6--13.4 ms on XLA-ROCm and 10.3--61.5 ms on
IREE-ROCm against 43--965 ms on the numpy reference interpreter, for the six
shipped benchmark cases at seed 42; the whole matrix -- compile times, the
generations needed to amortise them, and the numerical cross-check -- is
``benchmarks/GPU.md``.

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
