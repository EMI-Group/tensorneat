"""Host-side helpers over a gene payload: config rendering, batch chunking, arity checks.

Everything here is host-only Python, but some of it runs *inside* an ``evoxc``
trace: :func:`_jsonable` and friends render configuration values for ``config.txt``;
the tiling helpers -- :func:`_slice_batch` and :func:`_run_network` -- are the body of
the ``act_fn`` the pipeline hands a problem, i.e. plain Python that emits ops into the
trace that calls it and walks a batch-first payload only along the leaves a forward
pass reads; :func:`_one_individual` normalises a payload to its four batch-first arrays.

The tiled batch itself is *not* built here: a problem hands the network ``P * S`` rows
while the schedule and the payload stay at ``P``, and the genome expands the difference
internally (:func:`tensorneat.genome.window.by_samples`), because every operand a forward
derives is sample-invariant.
"""

from __future__ import annotations

from typing import Any, Callable, Optional

import evoxc
import numpy as np

from .common import State, TensorNEATError, to_numpy
from .genome.utils import GenomeTensors

__all__ = [
    "EVAL_CHUNK_ROWS",
    "_config_value",
    "_declared_shape",
    "_jsonable",
    "_one_individual",
    "_run_network",
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


def _jsonable(value: Any) -> Any:
    """Render one configuration value so ``json.dumps`` can always accept it.

    Primitives pass through (NumPy scalars unwrapped with ``.item()``),
    containers are rendered element-wise and anything else becomes
    ``repr(value)``, so a compiled executable or a gene object shows up in
    ``config.txt`` instead of aborting the checkpoint.
    """
    if value is None or isinstance(value, (bool, str)):
        return value
    if isinstance(value, (int, float)):
        return value
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_jsonable(item) for item in value]
    return repr(value)


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

    The static half of the chunked evaluate pass in :func:`_run_network`: a leaf
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


def _run_network(
    state: State,
    forward: Callable[[Any, Any, Any, Any], Any],
    transformed: Any,
    payload: GenomeTensors,
    inputs: Any,
) -> Any:
    """Evaluate a population on a tiled batch (the body of every ``act_fn``).

    ``transformed``/``payload`` arrive batch-first over ``P`` and stay there;
    ``inputs`` is ``(P * S, *input_shape)``.  The ``forward`` adapter
    (:meth:`Pipeline._forward_adapter`) is handed both at their native ``P`` rows and
    the tiled inputs as they arrive: the genome derives the sample count from the two
    static shapes and expands its own sample-invariant operands, so no repeat is
    emitted here and nothing has to be kept in step with a ``forward``'s read set.

    The one regrouping that remains is a chunk budget: a tiled block wider than
    :data:`EVAL_CHUNK_ROWS` rows is evaluated in groups of individuals whose own block
    fits that budget, and the group outputs are concatenated in row order.  This is
    bit-identical because every op of a ``forward`` is row-local -- ``gather`` /
    ``multiply`` / a reduce over the window axis / ``select`` / the elementwise
    ``output_transform`` -- and because the one population-wide quantity, the level
    ``while_loop``'s depth bound, is a *maximum*: a group's maximum never exceeds the
    global one and does bound each of its own rows, so the only level a row can miss is
    one beyond the static ``max_depth``, exactly as in the unchunked pass.

    The groups are a Python loop, so every group's entry-block intermediates stay
    alive until the generation returns; that is the interpreter's per-block live
    set, and nesting an ``evoxc.scan`` around the forward's level ``while_loop``
    would recover the peak but needs the control-flow nesting this package forbids
    -- only ``test/backends`` could prove it portable, so it stays out.

    Raises:
        TensorNEATError: If the batch is not a positive multiple of ``P``.
    """
    row_count = int(payload.node_keys.shape[0])
    batch = int(inputs.shape[0])
    factor, remainder = divmod(batch, row_count)
    if remainder or factor < 1:
        raise TensorNEATError(
            f"the problem asked the network for a batch of {batch} row(s), which is "
            f"not a positive multiple of the population {row_count}; a tiled "
            "dataset must be individual-major over the population axis"
        )
    chunk = (
        row_count
        if row_count * factor <= EVAL_CHUNK_ROWS
        else max(1, EVAL_CHUNK_ROWS // factor)
    )
    if chunk >= row_count:
        return forward(state, transformed, inputs, payload)
    outputs = []
    for start in range(0, row_count, chunk):
        stop = min(start + chunk, row_count)
        # Individual ``p`` owns the tiled rows ``p * factor .. (p + 1) * factor - 1``
        # (individual-major), so the group's rows are one contiguous block of ``inputs``.
        outputs.append(
            forward(
                state,
                _slice_batch(transformed, start, stop, row_count),
                inputs[start * factor : stop * factor],
                _slice_batch(payload, start, stop, row_count),
            )
        )
    return evoxc.concatenate(outputs, axis=0)


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
