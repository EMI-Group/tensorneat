"""Host-side helpers over a gene payload: config rendering, batch tiling, arity checks.

Everything here runs on the host, outside any ``evoxc`` trace.  :func:`_jsonable`
and friends render configuration values for ``config.txt``; :func:`_repeat_batch`
and :func:`_run_network` are the body of the ``act_fn`` the pipeline hands a
problem (they tile a batch-first payload over the tiling a problem uses);
:func:`_one_individual` normalises a payload to its four batch-first arrays.
"""

from __future__ import annotations

from typing import Any, Callable, Optional

import evoxc
import evoxc.numpy as enp
import numpy as np

from .common import State, TensorNEATError, to_numpy
from .genome.utils import GenomeTensors

__all__ = [
    "_config_value",
    "_declared_shape",
    "_jsonable",
    "_one_individual",
    "_repeat_batch",
    "_run_network",
    "_schedule_overflow",
]

#: The four payload field names, in ``GenomeTensors`` order.
_PAYLOAD_NAMES: tuple[str, ...] = ("node_keys", "node_attrs", "conn_keys", "conn_attrs")


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


def _repeat_batch(tree: Any, factor: int, pop_size: int) -> Any:
    """Repeat every batch-first leaf of a pytree ``factor`` times along axis 0.

    A problem tiles its batch over the population axis, so the schedule and the
    payload must be repeated to ``P * factor`` rows in the flat order
    ``p * factor + s`` (individual slowest) that the tiling uses.  Leaves
    without the population axis (a shared key or order table) pass through.
    """
    if factor == 1:
        return tree

    def expand(leaf: Any) -> Any:
        """Repeat one leaf along a new sample axis, if it carries the population."""
        shape = getattr(leaf, "shape", None)
        if not shape or int(shape[0]) != pop_size:
            return leaf
        trailing = tuple(int(size) for size in shape[1:])
        widened = enp.reshape(leaf, (pop_size, 1) + trailing)
        repeated = enp.broadcast_to(widened, (pop_size, factor) + trailing)
        return enp.reshape(repeated, (pop_size * factor,) + trailing)

    return evoxc.tree_map(expand, tree)


def _run_network(
    state: State,
    forward: Callable[[Any, Any, Any, Any], Any],
    transformed: Any,
    payload: GenomeTensors,
    inputs: Any,
) -> Any:
    """Evaluate a population on a tiled batch (the body of every ``act_fn``).

    ``transformed``/``payload`` arrive batch-first over ``P``; ``inputs`` is
    ``(P * S, *input_shape)``, so both are repeated ``S`` times before the
    ``forward`` adapter (:meth:`Pipeline._forward_adapter`) runs them.

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
    return forward(
        state,
        _repeat_batch(transformed, factor, row_count),
        inputs,
        _repeat_batch(payload, factor, row_count),
    )


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
