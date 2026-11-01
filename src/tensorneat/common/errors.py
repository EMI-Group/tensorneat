"""Host-side error hierarchy for TensorNEAT.

Dependency DAG position: leaf of ``tensorneat.common`` — this module imports
nothing from the package (no ``genome``, ``algorithm``, ``problem``,
``pipeline``) and nothing from ``jax``/``torch``.

Error policy (binding for the whole package):

* **Validation raises these, on the host, before a graph is traced.** A
  configuration mistake (two nodes with the same key, a key outside
  ``[0, KEY_MAX)``, a negative ``pop_size``, a missing activation name, ...)
  is reported with a :class:`TensorNEATError` subclass at build time, while
  a Python traceback can still point at the offending line of user code.
* **Backend and shape errors from ``evoxc`` are never caught, wrapped, or
  re-raised.** ``evoxc.BackendError``, ``evoxc.ShapeError``,
  ``evoxc.TraceError``, ... propagate unchanged so the traceback names the
  kernel and the shape that disagreed; re-wrapping them would hide the
  backend's own diagnostic and make "never silently fall back to another
  backend" impossible to audit.
* **In-graph overflow is data, not an exception.** A schedule that cannot be
  represented in the static bounds reports it in
  :attr:`tensorization.Transformed.edge_overflow`; the host decides whether to
  rebuild (:class:`ShapeBoundError`) or to continue.
"""


class TensorNEATError(Exception):
    """Root of the TensorNEAT error hierarchy.

    Parameters
    ----------
    message : str, optional
        Human-readable explanation. Stored as the ``message`` attribute (in
        addition to ``Exception.args``) because ``Pipeline``/``show`` report
        it without unwrapping ``args``, and because subclasses share the same
        constructor.

    Attributes
    ----------
    message : str
        The explanation passed to the constructor; ``str(err)`` returns it.
    """

    def __init__(self, message: str = "") -> None:
        """Store ``message`` on the instance and hand it to ``Exception``.

        Args:
            message (str): Human-readable explanation; defaults to ``""`` so
                ``raise SomeTensorNEATError()`` stays legal.
        """
        self.message = message
        super().__init__(message)


class ConfigError(TensorNEATError):
    """A user-supplied configuration value violates the contract.

    Raised on the host at configuration/build time, for example when

    * two nodes of one genome carry the same gene key (keys must be unique),
    * a key falls outside ``[0, KEY_MAX)`` (``KEY_MAX = 2**24``, the bound the
      ``int32`` key columns and the parity ``float32`` spelling share — see
      :data:`tensorneat.common.tools.KEY_MAX`),
    * ``pop_size``/``max_nodes``/``max_conns``/``max_in_degree``/``key_span``
      is not a positive ``int``,
    * a genome class is used with an algorithm whose ``num_inputs`` /
      ``num_outputs`` disagree with it, or
    * a registry lookup (activation/aggregation name) misses.

    Never raised from inside an ``@evoxc.defn`` graph: a traced kernel cannot
    branch on host configuration.
    """


class ShapeBoundError(TensorNEATError):
    """A static schedule bound was exceeded and the graph must be rebuilt.

    The execution schedule
    (:func:`tensorneat.common.tensorization.derive_schedule`) is compiled for
    a fixed shape signature ``(pop_size, max_nodes, max_conns, max_in_degree,
    key_span)``. Exceeding one of those bounds is not a numerical failure but a
    re-compilation trigger:

    * ``max_in_degree`` — an evolved node gained more incoming edges than the
      schedule's ``K``; the kernel reports it as
      :attr:`~tensorneat.common.tensorization.Transformed.edge_overflow`
      and the host either rebuilds with a larger ``K`` or raises this error.
    * ``key_span`` — a gene key no longer fits the dense key→slot table.
    * ``max_conns`` — the connection array of a genome is full.

    Parameters are the same as :class:`TensorNEATError`.
    """


class GraphError(TensorNEATError):
    """The evolved graph is structurally invalid for the requested execution.

    Raised when a *feedforward* execution is required and the genome contains
    a cycle (``level`` cannot be assigned), or when a required input/output
    node key is missing from the connection payload. Recurrent genomes accept
    cycles by definition and therefore never raise this for cyclic topology.
    """


class BackendMismatchError(TensorNEATError):
    """Host state and compiled executable were built for different backends.

    Raised when, for example, a ``State`` produced under ``backend="iree"`` is
    resumed under ``backend="numpy"``, or when a cached executable is reused
    with a device that the backend that compiled it cannot serve. Real backend
    failures (an unsupported op, an unsupported shape) are **not** this error:
    they propagate as ``evoxc.BackendError`` unchanged.
    """
