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
A batch wider than :data:`EVAL_CHUNK_ROWS` is walked in row groups so the interpreter's
per-block live set stays bounded.
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
    "_NetworkActFn",
    "_config_value",
    "_declared_shape",
    "_jsonable",
    "_object_config",
    "_one_individual",
    "_prepared_query",
    "_schedule_overflow",
    "_slice_batch",
]

#: The four payload field names, in ``GenomeTensors`` order.
_PAYLOAD_NAMES: tuple[str, ...] = ("node_keys", "node_attrs", "conn_keys", "conn_attrs")

#: Target rows per tiled-evaluate chunk.  A problem tiles its dataset/rollout over the
#: population axis, so a ``CustomFuncFit(num_samples=500)`` generation at pop 500 asks for
#: 250 000 rows at once; slicing the batch into groups of at most this many rows keeps
#: each forward pass inside cache and caps the interpreter's per-block live set.
#: The value is the measured optimum on ``custom_pagie_p500`` (25 individuals per chunk);
#: the XOR/RL cases stay under it and take the unchunked path unchanged.
EVAL_CHUNK_ROWS = 12_500

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
    """Slice a batch-first tree to the individuals ``[start, stop)``.

    The static half of the chunked evaluate pass in :func:`_prepared_query`: a leaf
    carrying the population axis is sliced and a leaf shared by the whole population
    (a static ordering or key table, which has no such axis) comes through untouched.
    The bounds are static ints, so the slice is one graph op per leaf.  The walk is
    whole-tree rather than a declared read set because the slice is a *view* of an
    operand the caller already holds -- unlike a repeat, it neither derives nor copies
    anything a forward does not read.
    """

    def take(leaf: Any) -> Any:
        """Slice one leaf along its population axis, if it carries one."""
        shape = getattr(leaf, "shape", None)
        if not shape or int(shape[0]) != pop_size:
            return leaf
        return leaf[start:stop]

    return evoxc.tree_map(take, tree)


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
    """

    def __init__(
        self,
        source: Callable[[], tuple[State, Any, GenomeTensors]],
        forward: Callable[..., Any],
        prepare: Callable[..., Any],
    ) -> None:
        """Store the trace-time source and the algorithm's prepared-pass pair.

        Args:
            source: Zero-argument callable returning ``(state, transformed, payload)``;
                called once per :meth:`prepare`, always inside the active trace.
            forward: ``algorithm.forward_prepared``.
            prepare: ``algorithm.prepare_forward``.
        """
        self._source = source
        self._forward = forward
        self._prepare = prepare

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
        )


def _prepared_query(
    state: State,
    forward: Callable[..., Any],
    prepare: Callable[..., Any],
    transformed: Any,
    payload: GenomeTensors,
    batch_rows: int,
) -> Callable[[Any], Any]:
    """Prepare one forward pass for ``batch_rows`` tiled rows and return its query.

    ``transformed``/``payload`` arrive batch-first over ``P`` and stay there, the query
    is asked for ``P * S = batch_rows`` rows, and the genome derives ``S`` from the two
    static shapes and expands its own sample-invariant operands -- so no repeat is
    emitted here and nothing has to be kept in step with a ``forward``'s read set.

    The one regrouping that remains is a chunk budget: a tiled block wider than
    :data:`EVAL_CHUNK_ROWS` rows is evaluated in groups of individuals whose own block
    fits that budget, the group passes are prepared **once at trace time**, and the
    group outputs are concatenated in row order.  This is bit-identical because every
    op of a ``forward`` is row-local -- ``gather`` / ``multiply`` / a reduce over the
    window axis / ``select`` / the elementwise ``output_transform`` -- and because the
    one population-wide quantity, the level ``while_loop``'s depth bound, is a
    *maximum*: a group's maximum never exceeds the global one and does bound each of
    its own rows, so the only level a row can miss is one beyond the static
    ``max_depth``, exactly as in the unchunked pass.

    The groups are a Python loop, so every group's entry-block intermediates stay alive
    until the generation returns; that is the interpreter's per-block live set, and
    nesting an ``evoxc.scan`` around the forward's level ``while_loop`` would recover
    the peak but needs the control-flow nesting this package forbids -- only
    ``test/backends`` could prove it portable, so it stays out.

    Raises:
        TensorNEATError: If ``batch_rows`` is not a positive multiple of ``P``.
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

    chunk = (
        row_count
        if row_count * factor <= EVAL_CHUNK_ROWS
        else max(1, EVAL_CHUNK_ROWS // factor)
    )
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
