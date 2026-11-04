"""Per-trace cache of rank-0 graph constants, shared by the distance kernels.

``evoxc`` emits one ``constant`` op per ``enp.full`` call and never deduplicates them, so a
path that rebuilds the same handful of rank-0 scalars at every use site — and the distance
path inlines two gene families per call, ``2 * ceil(species_size / centres_per_call)`` times
per generation — spends a large share of its traced ops on identical constants.
A :class:`ConstantPool` is created once per top-level kernel entry (one ``prepare`` /
``measure`` / ``gene_measure`` call, or one whole generation when the caller threads it) and
updated in place, so each ``(dtype, value)`` pair costs exactly one op.

Only **signed** integer and floating dtypes live here: ``evoxc``'s ``iree`` backend declares
``float16/float32/float64/int8/int16/int32/int64/bool`` and nothing unsigned, so a ``uint32``
operand or constant makes that backend refuse the whole graph
(``BackendError: capability drift``).  The dense table's range test is therefore spelled in
``int32`` (two comparisons per key column) rather than in the unsigned one-comparison form.

The pool is per-trace state: a traced ``SymbolicTensor`` belongs to the builder that produced
it, so a pool must never outlive the trace it was created in and must never be module-global.
Reusing one tensor as several op operands is safe — it is a plain value with more than one
consumer — and it is what collapses the duplicates.

Example:
    >>> pool = ConstantPool()
    >>> one = pool.i32(1)
    >>> same = pool.i32(1)
    >>> one is same
    True
"""

import evoxc
import evoxc.numpy as enp

__all__ = ["ConstantPool"]


class ConstantPool:
    """Cache of the rank-0 constants of one trace, keyed by dtype and value.

    Attributes:
        _i32 (dict): ``int`` value to ``int32`` rank-0 tensor.
        _i64 (dict): ``int`` value to ``int64`` rank-0 tensor.
        _f32 (dict): ``repr`` of a float to ``float32`` rank-0 tensor.
    """

    __slots__ = ("_i32", "_i64", "_f32")

    def __init__(self):
        """Start an empty pool for one traced entry point."""
        self._i32 = {}
        self._i64 = {}
        self._f32 = {}

    def i32(self, value: int) -> evoxc.SymbolicTensor:
        """Return the pooled rank-0 ``int32`` constant ``value``.

        Args:
            value: Static integer baked into the trace.

        Returns:
            Rank-0 ``int32`` tensor, the same object on every call with the same ``value``.

        Notes:
            Spelled out instead of a bare Python literal because an ``int32`` tensor plus a
            Python ``int`` promotes to ``int64`` in ``evoxc``, which would widen every index
            computation of the kernels that use this.
        """
        tensor = self._i32.get(value)
        if tensor is None:
            tensor = enp.full((), value, dtype=evoxc.int32)
            self._i32[value] = tensor
        return tensor

    def i64(self, value: int) -> evoxc.SymbolicTensor:
        """Return the pooled rank-0 ``int64`` constant ``value``.

        Args:
            value: Static integer baked into the trace.

        Returns:
            Rank-0 ``int64`` tensor — the dtype of the packed key composites.
        """
        tensor = self._i64.get(value)
        if tensor is None:
            tensor = enp.full((), value, dtype=evoxc.int64)
            self._i64[value] = tensor
        return tensor

    def f32(self, value: float) -> evoxc.SymbolicTensor:
        """Return the pooled rank-0 ``float32`` constant ``value``.

        Args:
            value: Static float baked into the trace.

        Returns:
            Rank-0 ``float32`` tensor.

        Notes:
            Keyed by ``repr`` rather than by the float itself: ``0.0 == -0.0`` compares and
            hashes equal, so a value-keyed cache would silently hand out the wrong sign of
            zero.  ``repr`` distinguishes them and round-trips exactly.
        """
        key = repr(value)
        tensor = self._f32.get(key)
        if tensor is None:
            tensor = enp.full((), value, dtype=evoxc.float32)
            self._f32[key] = tensor
        return tensor
