"""Host-side helpers over a gene payload: config rendering, batch chunking, arity checks.

Everything here is host-only Python, but some of it runs *inside* an ``evoxc``
trace: :func:`_jsonable` and friends render configuration values for ``config.txt`` --
deterministically, so two identical runs write identical bytes and the file records the
search law (:func:`_object_config`); the query helpers -- :class:`_NetworkActFn`,
:func:`_prepared_query` and :func:`_slice_batch` -- are what the pipeline hands a problem
as ``act_fn``, i.e. plain Python that emits ops into the trace that calls it and walks a
batch-first payload only along the leaves a forward pass reads; :func:`_one_individual`
normalises a payload to its four batch-first arrays.

The tiled batch itself is *not* built here: a problem hands the network ``P * S`` rows
while the schedule and the payload stay at ``P``, and the genome expands the difference
internally (:func:`tensorneat.genome.window.by_samples`), because every operand a forward
derives is sample-invariant.
A batch wider than the chunk budget is walked in row groups so the interpreter's
per-block live set stays bounded; the budget is :data:`EVAL_CHUNK_ROWS` on the CPU
interpreter and the larger :func:`eval_chunk_rows` window cap on a non-CPU device,
where the unrolled group loop's *dispatch count* rather than its live set is the
cost.  The trees it slices are declared population-major, and a leaf whose
population axis cannot be established raises instead of being sliced silently.
"""

from __future__ import annotations

import dataclasses
import inspect
from typing import Any, Callable, Optional

import evoxc
import numpy as np

from .common import State, TensorNEATError, to_numpy
from .genome.utils import GenomeTensors

__all__ = [
    "EVAL_CHUNK_ROWS",
    "DEVICE_EVAL_CHUNK_BYTES",
    "_NetworkActFn",
    "_config_value",
    "_declared_shape",
    "_formulation_signature",
    "_formulation_statics",
    "_jsonable",
    "_object_config",
    "_one_individual",
    "_prepared_query",
    "_schedule_overflow",
    "_slice_batch",
    "eval_chunk_rows",
    "materialize_payload",
]

#: The four payload field names, in ``GenomeTensors`` order.
_PAYLOAD_NAMES: tuple[str, ...] = ("node_keys", "node_attrs", "conn_keys", "conn_attrs")

#: Target rows per tiled-evaluate chunk on a **CPU** device (``device=None`` or
#: ``"cpu"``).  A problem tiles its dataset/rollout over the population axis, so a
#: ``CustomFuncFit(num_samples=500)`` generation at pop 500 asks for 250 000 rows at
#: once; slicing the batch into groups of at most this many rows keeps each forward
#: pass inside cache and caps the interpreter's per-block live set.
#: The value is the measured optimum on ``custom_pagie_p500`` (25 individuals per
#: chunk); the XOR/RL cases stay under it and take the unchunked path unchanged.
EVAL_CHUNK_ROWS = 12_500

#: Per-chunk device-memory budget, in bytes, for a **non-CPU** device: one
#: ``(rows, max_nodes) float32`` forward window must stay under this, so a device
#: gets ``DEVICE_EVAL_CHUNK_BYTES // (4 * max_nodes)`` rows per chunk instead of
#: :data:`EVAL_CHUNK_ROWS`.  The CPU budget is a cache/live-set bound (the numpy
#: interpreter is memory-bandwidth bound), while a device's cost is *dispatch
#: sequence length*: the chunk loop is unrolled at trace time, so 20 chunks are 20
#: copies of the forward's dispatch sequence and a bigger tile is nearly free.
#: 1 GiB is an order-of-magnitude guard, not a measured optimum: the real window
#: also carries the ``max_in_degree`` axis and the interpreter's live set, so the
#: budget stays a *cap* and the shipped func-fit cases (250 000 rows at
#: ``max_nodes=50``, i.e. ~50 MB per level) take the single unchunked pass.
DEVICE_EVAL_CHUNK_BYTES = 1 << 30


def eval_chunk_rows(device_kind: Optional[str], max_nodes: int) -> int:
    """Rows per tiled-evaluate chunk for one device kind.

    The chunk budget is a *host-side* static: it decides at trace time how many
    unrolled group passes a tiled ``act_fn`` emits, so it is part of the compiled
    graph and belongs in the executable-cache signature
    (:meth:`tensorneat.pipeline.Pipeline._shape_signature`), never a per-run
    choice.

    Args:
        device_kind (str | None): ``.kind`` of the device the generation graph
            runs on -- ``None`` and ``"cpu"`` are the interpreter's own budget.
        max_nodes (int): The genome's static node-row count, i.e. the width of one
            ``(rows, max_nodes)`` forward window.

    Returns:
        int: :data:`EVAL_CHUNK_ROWS` on CPU; on a non-CPU device the larger row
        count that keeps one ``(rows, max_nodes) float32`` window under
        :data:`DEVICE_EVAL_CHUNK_BYTES`, never below the CPU budget (so the
        device leg is never *more* chunked than the CPU one).

    Notes:
        Read at trace time like :data:`EVAL_CHUNK_ROWS` itself, so a monkeypatched
        budget reaches both legs; a shipped func-fit batch (250 000 rows) fits
        inside the device budget and therefore becomes one unchunked pass.
    """
    if device_kind is None or str(device_kind) == "cpu":
        return EVAL_CHUNK_ROWS
    per_row = 4 * max(1, int(max_nodes))
    return max(EVAL_CHUNK_ROWS, int(DEVICE_EVAL_CHUNK_BYTES) // per_row)

#: How deep :func:`_object_config` follows nested records before emitting the class path.
_MAX_CONFIG_DEPTH = 6

#: Sentinel meaning "this value is not configuration data" (see :func:`_attribute_config`).
_OMIT = object()


def _is_function_like(value: Any) -> bool:
    """Whether ``value`` is a function/class (named) rather than a callable object.

    The distinction matters: ``DefaultMutation`` and friends implement ``__call__``,
    so a bare ``callable()`` test would render an operator's *whole configuration* as
    one name string.  Only the things that genuinely have no state -- functions,
    lambdas, builtins, methods, ufuncs and classes -- are rendered by name.
    """
    return (
        isinstance(value, type)
        or inspect.isroutine(value)
        or isinstance(value, np.ufunc)
    )


def _qualified_name(value: Any) -> str:
    """The ``module.qualname`` path of a callable, class or object -- never an address.

    Plain functions, lambdas, ufuncs, bound methods and classes all render this way,
    so the same configuration always spells the same name.  An object without a name
    of its own (a ``functools.partial``, say) falls back to its type's path, which is
    equally deterministic.
    """
    module = getattr(value, "__module__", None) or getattr(
        type(value), "__module__", ""
    )
    qualname = getattr(value, "__qualname__", None) or getattr(value, "__name__", None)
    if not qualname:
        owner = type(value)
        return f"{owner.__module__}.{owner.__qualname__}"
    return f"{module}.{qualname}" if module else qualname


def _public_fields(value: Any) -> dict[str, Any]:
    """The public instance attributes of ``value`` as a name -> value mapping.

    Instance state only (never the methods ``dir()`` would add), across both
    ``__dict__`` and ``__slots__``, with private names dropped and the result sorted
    so the rendering is byte-stable.
    """
    fields: dict[str, Any] = {}
    for klass in type(value).__mro__:
        slots = klass.__dict__.get("__slots__", ())
        if isinstance(slots, str):
            slots = (slots,)
        for name in slots:
            if name.startswith("_"):
                continue
            try:
                fields[name] = getattr(value, name)
            except AttributeError:
                continue
    for name, item in getattr(value, "__dict__", {}).items():
        if not name.startswith("_"):
            fields[name] = item
    return dict(sorted(fields.items()))


def _object_config(value: Any, depth: int = 0) -> dict[str, Any]:
    """Render one configuration object as a deterministic mapping.

    The mapping is ``{"class": "module.qualname"}`` plus every public instance
    attribute whose value is configuration data: a primitive, a NumPy scalar, a
    container of those (recursively), a callable rendered by name, or a
    ``dataclasses``-style record rendered by the same rules.  Anything else -- a
    nested arbitrary object, an array, an ``evoxc`` tensor -- is omitted rather than
    rendered as a ``repr``, so no memory address can reach ``config.txt``.

    Args:
        value (Any): The object to describe (a gene, an operator, a record).
        depth (int): Current nesting depth; at :data:`_MAX_CONFIG_DEPTH` the class
            path is emitted alone, which bounds a pathological object graph.

    Returns:
        dict: The class path, followed by the sorted (name, rendered value) pairs.
    """
    rendered: dict[str, Any] = {"class": _qualified_name(type(value))}
    if depth >= _MAX_CONFIG_DEPTH:
        return rendered
    for name, attribute in _public_fields(value).items():
        item = _attribute_config(attribute, depth + 1)
        if item is not _OMIT:
            rendered[name] = item
    return rendered


def _attribute_config(value: Any, depth: int) -> Any:
    """Render one attribute value, or return ``_OMIT`` when it is not config data.

    This is the recursion step shared by :func:`_jsonable` (depth 0) and
    :func:`_object_config` (an object's public attributes), so an attribute and a
    top-level value are rendered by exactly the same rules.
    """
    if isinstance(value, np.generic):
        return value.item()
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if _is_function_like(value):
        return _qualified_name(value)
    if isinstance(value, dict):
        rendered_dict: dict[str, Any] = {}
        for key, item in value.items():
            rendered_item = _attribute_config(item, depth + 1)
            if rendered_item is not _OMIT:
                rendered_dict[str(key)] = rendered_item
        return rendered_dict
    if isinstance(value, (list, tuple, set, frozenset)):
        rendered_items: list[Any] = []
        for item in value:
            rendered_item = _attribute_config(item, depth + 1)
            if rendered_item is not _OMIT:
                rendered_items.append(rendered_item)
        if isinstance(value, (set, frozenset)):
            # Set iteration order is not reproducible across processes.
            rendered_items.sort(key=repr)
        return rendered_items
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return _object_config(value, depth)
    return _OMIT



def _jsonable(value: Any) -> Any:
    """Render one configuration value so ``json.dumps`` can always accept it.

    Primitives pass through (NumPy scalars unwrapped with ``.item()``), containers
    are rendered element-wise, a callable becomes its ``module.qualname`` path and
    any other object becomes the deterministic mapping :func:`_object_config`
    builds.  Nothing is ever rendered as a ``repr``, so ``config.txt`` names the
    gene laws, the gates and the transforms that produced the log, and two
    identical runs write identical bytes.
    """
    rendered = _attribute_config(value, 0)
    if rendered is _OMIT:
        return _object_config(value, 0)
    return rendered


def _config_value(owner: Any, name: str) -> Any:
    """The ``_jsonable`` value of ``owner.name``, or ``None`` when it does not exist.

    Only the two "this value is not available here" errors (``AttributeError``,
    ``NotImplementedError`` from an abstract shape) are absorbed; any other
    exception is a bug in the configuration object and propagates.
    """
    try:
        value = getattr(owner, name)
    except (AttributeError, NotImplementedError):
        return None
    return _jsonable(value)


def _declared_shape(owner: Any, name: str) -> Optional[tuple[int, ...]]:
    """The per-sample ``input_shape``/``output_shape`` of ``owner``, or ``None``.

    Raises:
        TensorNEATError: If the declared value is not a sequence of ints -- a
            broken declaration must not read as "no shape".
    """
    try:
        value = getattr(owner, name)
    except (AttributeError, NotImplementedError):
        return None
    try:
        return tuple(int(size) for size in value)
    except TypeError as error:
        raise TensorNEATError(
            f"{type(owner).__name__}.{name} must be a sequence of ints, got {value!r}"
        ) from error


def _schedule_overflow(schedule: Any) -> Any:
    """The ``(P,)`` overflow counter of a schedule, or ``None`` if it has none.

    A genome ``Transformed`` exposes ``edge_overflow`` directly; a wrapper such
    as ``HyperNEAT``'s ``SubstrateSchedule`` nests the genome schedule in its
    ``schedule`` field; a fixed topology overflows by construction, never.
    The branch is over *types* at trace time, never over a traced value.
    """
    overflow = getattr(schedule, "edge_overflow", None)
    if overflow is None:
        overflow = getattr(getattr(schedule, "schedule", None), "edge_overflow", None)
    return overflow


def _slice_batch(tree: Any, start: int, stop: int, pop_size: int) -> Any:
    """Slice a **declared population-major** tree to the individuals ``[start, stop)``.

    The static half of the chunked evaluate pass in :func:`_prepared_query`.  The
    trees that reach it are declared population-major by the kernel contract: a
    genome's ``Transformed`` schedule and the four-array gene payload both put
    the population on axis 0 (``src (P, C)``, ``in_conn (P, N, K)``,
    ``node_keys (P, N)``, ...), whatever the algorithm wraps around them
    (``HyperNEAT``'s ``SubstrateSchedule`` holds the substrate schedule and its
    ``(P, N)``/``(P, Q, ...)`` payload, all population-major).

    The declaration is what makes the slice unambiguous, and it is enforced
    rather than guessed: inside such a tree a non-scalar leaf **must** carry
    ``pop_size`` on axis 0, so a leaf whose leading dimension differs is not a
    population-major tensor (or the tree broke its own contract) and raises
    :class:`~tensorneat.common.errors.TensorNEATError` instead of being silently
    passed through or sliced.  A shape-only check cannot tell a population row
    from a shared table of coincidentally equal length and would mis-evaluate the
    group; only rank-0 leaves (a scalar, no axis at all) are shared through
    untouched.  Bounds are static ints, so the slice is one graph op per leaf.

    Args:
        tree (Any): A population-major pytree (schedule or gene payload).
        start (int): First individual of the group (static).
        stop (int): One past the last individual of the group (static).
        pop_size (int): Static population size that every leaf's axis 0 must
            equal.

    Returns:
        Any: The same tree structure with every leaf sliced along axis 0.

    Raises:
        TensorNEATError: If a non-scalar leaf's leading dimension is not
            ``pop_size``: its population axis cannot be established, and a
            chunk group must never be evaluated against a mis-sliced operand.
    """

    def take(leaf: Any) -> Any:
        """Slice one declared population-major leaf along its population axis."""
        shape = tuple(getattr(leaf, "shape", ()))
        if not shape:
            # A rank-0 leaf carries no axis at all, so there is nothing to slice
            # and nothing that a group could shift.
            return leaf
        leading = int(shape[0])
        if leading != pop_size:
            raise TensorNEATError(
                f"cannot chunk the tiled evaluate pass: a leaf of shape {shape} "
                "does not carry "
                f"the population axis (leading dim {leading} != pop_size "
                f"{pop_size}); the chunked evaluate pass slices only declared "
                "population-major tensors, and a leaf whose axis cannot be "
                "established must never be sliced or shared silently"
            )
        return leaf[start:stop]

    return evoxc.tree_map(take, tree)


def materialize_payload(payload: Any) -> GenomeTensors:
    """Host numpy arrays for a payload a backend may still hold on the device.

    :meth:`tensorneat.pipeline.Pipeline.step` hands the evaluated payload back
    **as the backend left it** when the run was built with
    ``Pipeline(host_payload=False)`` -- numpy on the reference interpreter,
    device-resident ``evoxc.Tensor`` leaves on a device backend -- so every
    reporting path goes through this helper to materialise it, and a caller that
    opted out can use it on the payload it received.  Materialisation is
    :func:`~tensorneat.common.tools.to_numpy`, the single explicit
    device-to-host step, applied per field; a payload that is already host numpy
    passes through untouched, which is the reference interpreter's case.

    Args:
        payload (Any): A ``GenomeTensors`` (or any four-element sequence) with the
            population on axis 0, or ``None`` when the step was told not to
            materialise it.

    Returns:
        GenomeTensors: Four host numpy arrays, population on axis 0.

    Raises:
        TensorNEATError: If ``payload`` is ``None``: a report that needs the host
            payload cannot be produced without it, and saying so names the flag
            that turned it off instead of reporting nothing.
    """
    if payload is None:
        raise TensorNEATError(
            "this report needs the host gene payload of the generation just "
            "stepped, but the step was told not to materialise it "
            "(Pipeline(host_payload=False)). Build the pipeline with "
            "host_payload=True (the default) when checkpoints, best-genome "
            "tracking or show_details are wanted, or call "
            "step(state, host_payload=True) for the generations that need it"
        )
    fields = tuple(payload)
    if all(isinstance(field, np.ndarray) for field in fields):
        return payload if isinstance(payload, GenomeTensors) else GenomeTensors(*fields)
    return GenomeTensors(*(np.asarray(to_numpy(field)) for field in fields))


def _finite_fitness(fitnesses: Any) -> np.ndarray:
    """A host view of ``fitnesses`` with every non-finite entry set to ``-inf``.

    ``algorithm.tell`` already sanitises its own input inside the graph
    (``algorithm/neat/neat.py::_sanitize_fitness`` maps ``NaN`` *and* ``±inf`` to
    ``-inf``) and every shipped problem ends with
    ``problem.base.finalize_fitness``, which maps ``NaN`` only.  A custom problem
    can therefore still hand a non-finite fitness to the host, where ``+inf``
    would falsely satisfy ``Pipeline.fitness_target`` and ``NaN`` would win
    ``argmax``.  This is the one host-side view that closes that gap, and it is
    the identity on every recorded run (shipped fitnesses are finite or
    ``-inf``).

    Args:
        fitnesses (Any): ``(P,)`` host fitness as ``Pipeline.step`` returns it.

    Returns:
        np.ndarray: ``(P,)`` with each ``NaN``/``+inf``/``-inf`` entry replaced by
        ``-inf`` and every finite entry passed through bit-identically.
    """
    values = np.asarray(fitnesses)
    return np.where(np.isfinite(values), values, -np.inf)


#: Device kinds whose kernels take the GPU-friendly formulation: the GPU admits
#: wide batches far better than it admits the merge/sort expansion of the
#: compatibility distance, so these kinds ask for an index-free kernel over the
#: population-shaped centre batch.
_GPU_DEVICE_KINDS: tuple[str, ...] = ("rocm", "hip", "hips", "cuda", "vulkan", "gpu")

#: The three trace-time statics a GPU-class kind registers, in registration order:
#: ``(distance_centre, distance_formulation, kernel_mode)``.
_GPU_FORMULATION: tuple[str, str, str] = ("tiled", "index_free", "accelerator")

#: The triple every other kind registers -- the host leg where the merge kernel is
#: the measured winner, one centre per call keeps element volume down and every
#: kernel takes its legacy formulation.
_HOST_FORMULATION: tuple[str, str, str] = ("single", "merge", "host")

#: What a *missing* formulation static means: the kernels themselves fall back to
#: these values, so a hand-built ``State`` (one no ``Pipeline.setup`` produced)
#: keeps the legacy contract.
_DEFAULT_DISTANCE_CENTRE = "tiled"
_DEFAULT_DISTANCE_FORMULATION = "merge"
_DEFAULT_KERNEL_MODE = "host"


def _device_kind(device: Any) -> Optional[str]:
    """The resolved *kind* of ``device``, spelled the way ``evoxc`` resolves it.

    ``Pipeline._placement_device`` maps ``None`` to "no device", a kind string to
    ``evoxc.Device(kind)`` and an ``evoxc.Device`` to itself; the kind is
    therefore the string itself or the device's ``kind`` field.  A value carrying
    no string kind is reported as ``None`` (the non-GPU default) instead of
    raising: the placement itself hands such a value to ``evoxc``'s own
    ``Tensor.to``, whose ``TypeError`` names the real problem.
    """
    if device is None:
        return None
    if isinstance(device, str):
        return device
    kind = getattr(device, "kind", None)
    return kind if isinstance(kind, str) else None


def _formulation_statics(device: Any) -> tuple[str, str, str]:
    """The trace-time kernel statics ``device`` selects, as one triple.

    These are trace-time kernel selections, not tensors: they are registered on the
    state as plain strings and baked into the generation graph by the pipeline's
    static-config pass.  They all follow from one decision -- whether the resolved
    kind is GPU-class (:data:`_GPU_DEVICE_KINDS`) -- so they are derived here once
    rather than by each kernel table lookup: a GPU-class kind gets
    :data:`_GPU_FORMULATION` (index-free distance over the population-shaped centre
    batch, accelerator formulations elsewhere), every other kind -- including
    ``None`` -- :data:`_HOST_FORMULATION`.  The comparison is case-insensitive
    because ``evoxc`` stores a kind string verbatim.

    Args:
        device (Any): The pipeline's device selection: ``None``, a kind string or
            an ``evoxc.Device``.

    Returns:
        tuple: ``(distance_centre, distance_formulation, kernel_mode)`` --
        ``("tiled" | "single", "index_free" | "merge", "accelerator" | "host")``.
    """
    kind = _device_kind(device)
    if isinstance(kind, str) and kind.lower() in _GPU_DEVICE_KINDS:
        return _GPU_FORMULATION
    return _HOST_FORMULATION


def _formulation_signature(state: Any) -> tuple[str, str, str]:
    """The formulation statics of ``state``, defaulted to the legacy contract.

    Part of the executable-cache key: the triple changes which kernels the graph
    formulates, so reusing an executable across a device/mode change would run the
    wrong formulation.  A state no ``Pipeline.setup`` produced falls back to the
    same values the kernels themselves fall back to
    (:data:`_DEFAULT_DISTANCE_CENTRE` / :data:`_DEFAULT_DISTANCE_FORMULATION` /
    :data:`_DEFAULT_KERNEL_MODE`).

    Args:
        state (Any): A state carrying the statics, or any object without them.

    Returns:
        tuple: ``(distance_centre, distance_formulation, kernel_mode)``.
    """
    return (
        getattr(state, "distance_centre", _DEFAULT_DISTANCE_CENTRE),
        getattr(state, "distance_formulation", _DEFAULT_DISTANCE_FORMULATION),
        getattr(state, "kernel_mode", _DEFAULT_KERNEL_MODE),
    )


def _substrate_signature(algorithm: Any) -> tuple[Any, ...]:
    """The static geometry of ``algorithm.substrate``, or ``()`` with none.

    A ``HyperNEAT`` generation traces the substrate's node/connection counts, its
    coordinate and endpoint tables and the ``activate_time`` horizon into the
    graph, and bakes the CPPN-to-weight map's thresholds as constants.  All of
    them therefore belong in the pipeline's executable-cache key: the substrate is
    host-only (it is never part of ``State``), so a caller can replace it on a
    live algorithm, and reusing an executable built for the old geometry would
    evaluate the wrong phenotype.

    Args:
        algorithm (Any): The algorithm being driven; a non-wrapper one has no
            ``substrate`` attribute and contributes nothing.

    Returns:
        tuple: ``(nodes_cnt, conns_cnt, num_inputs, num_outputs, activate_time,
        nodes.shape, conn_index.shape, query_coors.shape, weight_threshold,
        max_weight)``, or ``()`` when the algorithm wraps no substrate.
    """
    substrate = getattr(algorithm, "substrate", None)
    if substrate is None:
        return ()

    def count(name: str) -> int:
        """Read one substrate static, ``-1`` when the substrate lacks it."""
        value = getattr(substrate, name, None)
        return -1 if value is None else int(value)

    def table(name: str) -> tuple[int, ...]:
        """The static shape of one substrate host table (``()`` if absent)."""
        return tuple(int(size) for size in np.shape(getattr(substrate, name, ())))

    return (
        count("nodes_cnt"),
        count("conns_cnt"),
        count("num_inputs"),
        count("num_outputs"),
        int(getattr(algorithm, "activate_time", -1)),
        table("nodes"),
        table("conn_index"),
        table("query_coors"),
        float(getattr(algorithm, "weight_threshold", float("nan"))),
        float(getattr(algorithm, "max_weight", float("nan"))),
    )


class _NetworkActFn:
    """The ``act_fn`` a problem receives: one query, or one pass prepared per batch shape.

    A problem that queries one network many times -- a rollout calls the policy once
    per timestep -- calls :meth:`prepare` **before** its loop and queries the returned
    callable inside it, so the loop-invariant genome prelude is derived once per
    generation instead of once per timestep.
    ``act_fn(inputs)`` stays the one-shot path (func fit, ``show``): it prepares a pass
    for exactly that batch and queries it once, i.e. the pre-hoisting behaviour.

    The collaborator triple is injected, so the class stays algorithm-agnostic:
    ``source()`` returns ``(state, transformed, payload)`` inside the active trace,
    ``forward`` is ``algorithm.forward_prepared`` and ``prepare`` is
    ``algorithm.prepare_forward``.

    ``device_kind`` is the *static* ``.kind`` of the device the surrounding graph
    runs on, so :func:`_prepared_query` can pick the chunk budget at trace time
    (:func:`eval_chunk_rows`).  It stays ``None`` for a host-reporting
    ``act_fn``, which always evaluates on the CPU side-channel.
    """

    def __init__(
        self,
        source: Callable[[], tuple[State, Any, GenomeTensors]],
        forward: Callable[..., Any],
        prepare: Callable[..., Any],
        device_kind: Optional[str] = None,
    ) -> None:
        """Store the trace-time source and the algorithm's prepared-pass pair.

        Args:
            source: Zero-argument callable returning ``(state, transformed, payload)``;
                called once per :meth:`prepare`, always inside the active trace.
            forward: ``algorithm.forward_prepared``.
            prepare: ``algorithm.prepare_forward``.
            device_kind (str | None, optional): ``.kind`` of the device the graph
                this ``act_fn`` belongs to runs on (``None`` = CPU).  Defaults to
                ``None``.
        """
        self._source = source
        self._forward = forward
        self._prepare = prepare
        self._device_kind = device_kind

    def __call__(self, inputs: Any) -> Any:
        """Query the network once on ``inputs``, preparing a pass for exactly its batch.

        Args:
            inputs: ``(P * S, *input_shape)`` input activations, tiled individual-major.

        Returns:
            ``(P * S, *output_shape)`` outputs.
        """
        return self.prepare(int(inputs.shape[0]))(inputs)

    def prepare(self, batch_rows: int) -> Callable[[Any], Any]:
        """Prepare one forward pass for a tiled batch of ``batch_rows`` rows.

        Called once per generation by a problem that queries the network repeatedly;
        the returned query may then be called any number of times on inputs of exactly
        that row count.

        Args:
            batch_rows (int): Rows the pass will be queried with, ``P * S``.

        Returns:
            Callable: ``query(inputs) -> outputs`` for that one batch shape.
        """
        state, transformed, payload = self._source()
        return _prepared_query(
            state,
            self._forward,
            self._prepare,
            transformed,
            payload,
            int(batch_rows),
            device_kind=self._device_kind,
        )


def _prepared_query(
    state: State,
    forward: Callable[..., Any],
    prepare: Callable[..., Any],
    transformed: Any,
    payload: GenomeTensors,
    batch_rows: int,
    device_kind: Optional[str] = None,
) -> Callable[[Any], Any]:
    """Prepare one forward pass for ``batch_rows`` tiled rows and return its query.

    ``transformed``/``payload`` arrive batch-first over ``P`` and stay there, the query
    is asked for ``P * S = batch_rows`` rows, and the genome derives ``S`` from the two
    static shapes and expands its own sample-invariant operands -- so no repeat is
    emitted here and nothing has to be kept in step with a ``forward``'s read set.

    The one regrouping that remains is a chunk budget: a tiled block wider than
    :func:`eval_chunk_rows` rows is evaluated in groups of individuals whose own block
    fits that budget, the group passes are prepared **once at trace time**, and the
    group outputs are concatenated in row order.  This is bit-identical because every
    op of a ``forward`` is row-local -- ``gather`` / ``multiply`` / a reduce over the
    window axis / ``select`` / the elementwise ``output_transform`` -- and because the
    one population-wide quantity, the level ``while_loop``'s depth bound, is a
    *maximum*: a group's maximum never exceeds the global one and does bound each of
    its own rows, so the only level a row can miss is one beyond the static
    ``max_depth``, exactly as in the unchunked pass.

    ``device_kind`` selects that budget at trace time, because the budget is a
    *compiled* property: on the CPU interpreter the groups bound the per-block live
    set (memory bandwidth is the cost), so :data:`EVAL_CHUNK_ROWS` stays exactly the
    measured optimum; on a non-CPU device the unrolled group loop is a *dispatch
    sequence*, so the budget becomes the larger :func:`eval_chunk_rows` window cap
    and a shipped func-fit batch is one unchunked pass.

    The groups are a Python loop, so every group's entry-block intermediates stay alive
    until the generation returns; that is the interpreter's per-block live set, and
    nesting an ``evoxc.scan`` around the forward's level ``while_loop`` would recover
    the peak but needs the control-flow nesting this package forbids -- only
    ``test/backends`` could prove it portable, so it stays out.

    Both trees handed to :func:`_slice_batch` are declared population-major, so every
    non-scalar leaf must carry ``pop_size`` on axis 0; a leaf whose axis cannot be
    established raises there instead of being silently shared or mis-sliced.

    Args:
        state (State): Configuration state of the active trace.
        forward (Callable[..., Any]): ``algorithm.forward_prepared``.
        prepare (Callable[..., Any]): ``algorithm.prepare_forward``.
        transformed (Any): The population-major schedule at ``P`` rows.
        payload (GenomeTensors): The population-major payload at ``P`` rows.
        batch_rows (int): Tiled rows the returned query will be asked for, ``P * S``.
        device_kind (str | None, optional): ``.kind`` of the device the graph runs
            on (``None`` = CPU).  Defaults to ``None``.

    Raises:
        TensorNEATError: If ``batch_rows`` is not a positive multiple of ``P``, or if a
            leaf of the schedule/payload cannot be established as population-major.
    """
    row_count = int(payload.node_keys.shape[0])
    batch = int(batch_rows)
    factor, remainder = divmod(batch, row_count)
    if remainder or factor < 1:
        raise TensorNEATError(
            f"the problem asked the network for a batch of {batch} row(s), which is "
            f"not a positive multiple of the population {row_count}; a tiled "
            "dataset must be individual-major over the population axis"
        )

    def guard(inputs: Any) -> None:
        """Reject a query whose static row count is not the prepared one.

        Raises:
            TensorNEATError: If ``inputs`` does not carry ``batch_rows`` rows.
        """
        rows = int(inputs.shape[0])
        if rows != batch:
            raise TensorNEATError(
                f"this forward pass was prepared for {batch} row(s) but the query was "
                f"asked for {rows}; a prepared pass serves exactly one batch shape, so "
                "prepare a new pass for a different row count"
            )

    # ``max_nodes`` is the node-row width of one ``(rows, max_nodes)`` forward window,
    # i.e. what the device budget is expressed in.  Read at trace time so a
    # monkeypatched budget reaches both legs.
    budget = eval_chunk_rows(device_kind, int(payload.node_keys.shape[1]))
    chunk = row_count if row_count * factor <= budget else max(1, budget // factor)
    if chunk >= row_count:
        prepared = prepare(state, transformed, payload, factor)

        def query(inputs: Any) -> Any:
            """Evaluate the whole population on the prepared batch."""
            guard(inputs)
            return forward(state, transformed, prepared, inputs)

        return query

    groups = []
    for start in range(0, row_count, chunk):
        stop = min(start + chunk, row_count)
        # Individual ``p`` owns the tiled rows ``p * factor .. (p + 1) * factor - 1``
        # (individual-major), so the group's rows are one contiguous block of ``inputs``.
        group_transformed = _slice_batch(transformed, start, stop, row_count)
        group_payload = _slice_batch(payload, start, stop, row_count)
        groups.append(
            (
                start * factor,
                stop * factor,
                group_transformed,
                prepare(state, group_transformed, group_payload, factor),
            )
        )

    def query_chunked(inputs: Any) -> Any:
        """Evaluate every prepared group on its contiguous block of the batch."""
        guard(inputs)
        outputs = [
            forward(state, group_transformed, prepared, inputs[begin:end])
            for begin, end, group_transformed, prepared in groups
        ]
        return evoxc.concatenate(outputs, axis=0)

    return query_chunked


def _one_individual(payload: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Normalise a gene payload to its four batch-first arrays.

    The **batch-first** payload is the contract: one individual is a batch with a
    leading population axis of length 1 -- ``node_keys (1, max_nodes)``,
    ``node_attrs (1, max_nodes, A)``, ``conn_keys (1, max_conns, key_width)`` and
    ``conn_attrs (1, max_conns, W)``, i.e. exactly the spelling of ``state.pop``
    with ``P = 1``.

    Args:
        payload (Any): A ``GenomeTensors`` (or any four-element sequence) written
            batch-first.  The bare per-individual rows a checkpoint stores
            (``node_keys (N,)``, ``node_attrs (N, A)``, ``conn_keys (E, CK)``,
            ``conn_attrs (E, CA)``) are also accepted and promoted with a leading
            axis of 1, so a hand-built host row can be reported with
            :meth:`Pipeline.show`.

    Returns:
        tuple: ``(node_keys, node_attrs, conn_keys, conn_attrs)`` as numpy arrays
        shaped ``(P, N)``, ``(P, N, A)``, ``(P, E, CK)``, ``(P, E, CA)`` with the
        one shared population axis ``P`` (``P = 1`` for bare input).

    Raises:
        TensorNEATError: If the payload is missing (``None``), is not four
            non-scalar arrays, is a batch of more than one individual
            (``node_keys.shape[0] > 1`` -- this helper normalises *one* genome),
            if a field is written with a rank the payload cannot have
            (``node_keys`` above rank 2, ``conn_keys`` outside rank 2-3, an
            attribute array whose rank does not match the key array's spelling),
            or if the four arrays disagree about the population axis.  A
            population is never reshaped down to one individual.
    """
    if payload is None:
        raise TensorNEATError(
            "no genome payload was given (got None); pass a one-individual batch "
            "(1, max_nodes) -- the spelling of state.pop/best_genome at P = 1, "
            "e.g. the `best` that `auto_run(state)` returns -- or the bare "
            "per-individual rows a checkpoint stores: (max_nodes,), "
            "(max_nodes, attr_width), (max_conns, key_width), "
            "(max_conns, attr_width)"
        )
    fields = list(payload)
    if len(fields) != 4:
        raise TensorNEATError(
            "a genome payload is the four arrays (node_keys, node_attrs, conn_keys, "
            f"conn_attrs), got {len(fields)} array(s) of {type(payload).__name__}"
        )
    arrays = []
    for index, field in enumerate(fields):
        array = np.asarray(to_numpy(field))
        if array.ndim == 0:
            raise TensorNEATError(
                f"payload field {index} ({_PAYLOAD_NAMES[index]}) is rank-0; "
                "expected a batch-first array (node_keys (P, max_nodes), node_attrs "
                "(P, max_nodes, A), conn_keys (P, max_conns, key_width), conn_attrs "
                "(P, max_conns, W)) or the bare per-individual rows"
            )
        arrays.append(array)
    node_keys, node_attrs, conn_keys, conn_attrs = arrays

    # ``node_keys`` decides the spelling: a bare row is rank 1 ``(max_nodes,)``,
    # a batch-first payload is rank 2 ``(P, max_nodes)``.  Never rank 3+.
    if node_keys.ndim > 2:
        raise TensorNEATError(
            f"payload field node_keys is rank-{node_keys.ndim} with shape "
            f"{node_keys.shape}; expected the batch-first (P, max_nodes) or the "
            "bare (max_nodes,) row"
        )
    if conn_keys.ndim not in (2, 3):
        raise TensorNEATError(
            f"payload field conn_keys is rank-{conn_keys.ndim} with shape "
            f"{conn_keys.shape}; expected the batch-first (P, max_conns, key_width) "
            "or the bare (max_conns, key_width) row"
        )

    batched = node_keys.ndim == 2
    if batched:
        expected = {
            "node_keys": "(P, max_nodes)",
            "node_attrs": "(P, max_nodes, attr_width)",
            "conn_keys": "(P, max_conns, key_width)",
            "conn_attrs": "(P, max_conns, attr_width)",
        }
        ranks = {"node_keys": 2, "node_attrs": 3, "conn_keys": 3, "conn_attrs": 3}
    else:
        expected = {
            "node_keys": "(max_nodes,)",
            "node_attrs": "(max_nodes, attr_width)",
            "conn_keys": "(max_conns, key_width)",
            "conn_attrs": "(max_conns, attr_width)",
        }
        ranks = {"node_keys": 1, "node_attrs": 2, "conn_keys": 2, "conn_attrs": 2}

    for name, array in zip(_PAYLOAD_NAMES, arrays):
        if array.ndim != ranks[name]:
            raise TensorNEATError(
                f"payload field {name} is rank-{array.ndim} with shape {array.shape}; "
                f"expected {expected[name]} (the four fields are either all "
                "batch-first or all bare per-individual rows)"
            )

    if batched:
        # The batch is the *one-individual* batch: a whole population (P > 1) is
        # a usage error, reported here instead of by the population-axis check
        # below, whose message would blame an axis disagreement that does not
        # exist.
        population = int(node_keys.shape[0])
        if population != 1:
            raise TensorNEATError(
                f"the payload holds {population} individuals (node_keys shape "
                f"{node_keys.shape}); this helper normalises one genome, so pass "
                "a one-individual batch (1, max_nodes) -- node_keys "
                "(1, max_nodes), node_attrs (1, max_nodes, attr_width), conn_keys "
                "(1, max_conns, key_width), conn_attrs (1, max_conns, "
                "attr_width), i.e. state.pop/best_genome at P = 1 -- or the bare "
                "per-individual rows (max_nodes,), (max_nodes, attr_width), "
                "(max_conns, key_width), (max_conns, attr_width)"
            )
        leaders = {name: int(array.shape[0]) for name, array in zip(_PAYLOAD_NAMES, arrays)}
        if len(set(leaders.values())) != 1:
            raise TensorNEATError(
                "the four payload arrays disagree about the population axis: "
                f"node_keys {node_keys.shape}, node_attrs {node_attrs.shape}, "
                f"conn_keys {conn_keys.shape}, conn_attrs {conn_attrs.shape}; "
                "expected (P, max_nodes), (P, max_nodes, attr_width), "
                "(P, max_conns, key_width), (P, max_conns, attr_width) with one "
                "shared P"
            )
        return node_keys, node_attrs, conn_keys, conn_attrs

    return tuple(array[np.newaxis, ...] for array in arrays)
