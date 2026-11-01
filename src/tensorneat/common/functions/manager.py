"""Host-side registry pairing evoxc kernels with their sympy counterparts.

``FunctionManager`` is a plain host object: it is built once at import time,
never traced, and holds two *ordered, same-length* tuples -- the evoxc kernels
of :mod:`.act` / :mod:`.agg` and the sympy ``sp.Function`` classes of
:mod:`.act_sympy` / :mod:`.agg_sympy`. The position of an entry is the value
stored in a node gene, so the order of both tuples is part of the gene
encoding.

The manager is immutable (no ``add_func`` / ``update_sympy``): a node gene holds
an index into *its own* ``activation_options`` / ``aggregation_options`` list,
so custom options are supplied per genome as plain callables and need no global
registration. A genome that wants sympy support for a custom option builds its
own ``FunctionManager`` and passes it around.

Errors raised here are programmer errors in a registry table (a mismatched
pairing, a duplicate name, a backend that no longer exists) and are therefore
plain ``ValueError``; user *configuration* errors belong to
``common.errors.ConfigError`` and are raised by the configuration layer.
"""

import re
from functools import partial
from typing import Any, Callable, Optional, Sequence, Union

import numpy as np

import evoxc


def _snake_case(name: str) -> str:
    """Convert a sympy class name to the snake_case registry name it must pair with.

    Parameters
    ----------
    name : str
        A sympy class name such as ``"SympyScaledSigmoid"``.

    Returns
    -------
    str
        ``"scaled_sigmoid"`` -- the ``Sympy`` prefix removed and every camel
        hump lowered.

    Notes
    -----
    Used only to validate the pairing, which is how ``"inv"`` can no longer be
    paired with ``SympyIdentity`` (the pre-rewrite bug).
    """
    snake = re.sub(r"(?<!^)(?=[A-Z])", "_", name).lower()
    prefix = "sympy_"
    return snake[len(prefix):] if snake.startswith(prefix) else snake


class FunctionManager:
    """Ordered (evoxc kernel, sympy counterpart) registry with name and id lookup.

    Parameters
    ----------
    evoxc_fns : sequence of callables
        Graph kernels, in canonical option order. Each one's ``__name__`` is
        the registry name, so kernels must be named after the option they
        implement (``scaled_sigmoid``, ``sum``, ...).
    sympy_fns : sequence of sympy ``sp.Function`` subclasses or ``None``
        Host-only counterparts, same length and same order as ``evoxc_fns``;
        ``None`` marks an option with no sympy representation.
    extra_sympy_classes : sequence of sympy ``sp.Function`` subclasses, optional
        Helper classes that appear inside generated expressions without being an
        option of their own (``SympyClip``, used by ``SympyExp``); they are
        emitted by :meth:`sympy_module` so ``sp.lambdify`` can evaluate them.

    Raises
    ------
    ValueError
        If a sequence is empty, a kernel is not callable, two kernels share a
        name, the two sequences differ in length, or a sympy class does not
        pair with the kernel at its position.

    Attributes
    ----------
    evoxc_fns : tuple of callables
        The kernels, in option order (this tuple is what the dispatchers
        default to).
    sympy_fns : tuple of sympy classes or ``None``
        The counterparts, in option order.
    extra_sympy_classes : tuple of sympy classes
        Codegen helpers that are not options themselves.
    names : tuple of str
        Registry names, in option order.
    index2name : tuple of str
        Alias of ``names``: ``index2name[id]`` is the name of option ``id``.
    name2index : dict of str to int
        Inverse of ``index2name``.

    Notes
    -----
    Attribute access resolves registry names (``ACT.sigmoid``, ``AGG.sum``) and
    is the canonical way to reference an option in a genome configuration;
    a name that collides with a real attribute is shadowed by the attribute and
    must be fetched with :meth:`get_evoxc_func` instead.
    """

    def __init__(
        self,
        evoxc_fns: Sequence[Callable],
        sympy_fns: Sequence[Optional[type]],
        extra_sympy_classes: Sequence[type] = (),
    ):
        evoxc_fns = tuple(evoxc_fns)
        sympy_fns = tuple(sympy_fns)
        if not evoxc_fns:
            raise ValueError("FunctionManager needs at least one evoxc kernel")
        for fn in evoxc_fns:
            if not callable(fn):
                raise ValueError(f"evoxc kernel {fn!r} is not callable")
        names = tuple(fn.__name__ for fn in evoxc_fns)
        if len(set(names)) != len(names):
            raise ValueError(f"duplicate option names in {names!r}")
        if len(sympy_fns) != len(evoxc_fns):
            raise ValueError(
                f"expected {len(evoxc_fns)} sympy counterparts, got {len(sympy_fns)}"
            )
        for name, sympy_cls in zip(names, sympy_fns):
            if sympy_cls is None:
                continue
            if _snake_case(getattr(sympy_cls, "__name__", "")) != name:
                raise ValueError(
                    f"sympy class {sympy_cls!r} does not pair with option {name!r}"
                )

        self.evoxc_fns = evoxc_fns
        self.sympy_fns = sympy_fns
        self.extra_sympy_classes = tuple(extra_sympy_classes)
        self.names = names
        self.index2name = names
        self.name2index = {name: i for i, name in enumerate(names)}

    def __len__(self) -> int:
        """Return the number of registered options.

        Returns
        -------
        int
            ``len(self.evoxc_fns)``.
        """
        return len(self.evoxc_fns)

    def __repr__(self) -> str:
        """Return a short host-side representation listing the option names.

        Returns
        -------
        str
            ``FunctionManager(scaled_sigmoid, sigmoid, ...)``.
        """
        return f"{type(self).__name__}({', '.join(self.names)})"

    def __getattr__(self, name: str) -> Callable:
        """Resolve a registry name to its evoxc kernel.

        Parameters
        ----------
        name : str
            Option name, e.g. ``"sigmoid"``.

        Returns
        -------
        Callable
            ``self.evoxc_fns[self.name2index[name]]``.

        Raises
        ------
        AttributeError
            If ``name`` is not a registered option (or starts with ``_``, so
            that ``__dict__`` access during copying/unpickling stays safe).
        """
        evoxc_fns = self.__dict__.get("evoxc_fns")
        name2index = self.__dict__.get("name2index")
        if name.startswith("_") or evoxc_fns is None or name not in name2index:
            raise AttributeError(f"{type(self).__name__!r} has no attribute {name!r}")
        return evoxc_fns[name2index[name]]

    def get_evoxc_func(self, index: Union[int, np.integer]) -> Callable:
        """Return the kernel stored at an option id.

        Parameters
        ----------
        index : int-like
            Option id -- a gene value read from the population, so numpy
            integers are accepted and coerced with ``int``.

        Returns
        -------
        Callable
            The kernel, suitable for the dispatchers' ``act_fns`` /
            ``agg_fns`` argument.

        Raises
        ------
        IndexError
            If ``index`` is outside ``[0, len(self))`` -- a negative id is
            rejected rather than wrapped, because a gene id that reaches here
            is either a valid option or a corrupted payload.
        """
        index = int(index)
        if not 0 <= index < len(self.evoxc_fns):
            raise IndexError(
                f"option id {index} is outside [0, {len(self.evoxc_fns)}) of {self!r}"
            )
        return self.evoxc_fns[index]

    def get_sympy_func(self, index: Union[int, np.integer]) -> type:
        """Return the sympy counterpart stored at an option id.

        Parameters
        ----------
        index : int-like
            Option id.

        Returns
        -------
        type
            A ``sp.Function`` subclass.

        Raises
        ------
        IndexError
            If ``index`` is outside ``[0, len(self))`` (negative ids are
            rejected, not wrapped).
        ValueError
            If the option has no sympy counterpart (``None`` entry).
        """
        index = int(index)
        if not 0 <= index < len(self.sympy_fns):
            raise IndexError(
                f"option id {index} is outside [0, {len(self.sympy_fns)}) of {self!r}"
            )
        sympy_cls = self.sympy_fns[index]
        if sympy_cls is None:
            raise ValueError(
                f"option {self.names[index]!r} has no sympy representation"
            )
        return sympy_cls

    def name_of(self, func: Callable) -> str:
        """Return the registry name of a kernel.

        Parameters
        ----------
        func : Callable
            A member of ``self.evoxc_fns`` (identity comparison -- a
            ``functools.partial`` or other wrapper is *not* recognised).

        Returns
        -------
        str
            The option name, e.g. ``"maxabs"``.

        Raises
        ------
        ValueError
            If ``func`` is not a registered kernel.

        Notes
        -----
        Replaces the removed module-level ``get_func_name`` helper: names are
        registry data now, not ``func.__name__`` string surgery.
        """
        for i, fn in enumerate(self.evoxc_fns):
            if fn is func:
                return self.names[i]
        raise ValueError(f"{func!r} is not registered in {self!r}")

    def obtain_sympy(self, func: Union[int, str, Callable]) -> type:
        """Return the sympy counterpart of an option, addressed flexibly.

        Parameters
        ----------
        func : int, str or Callable
            An option id, an option name, or a registered kernel.

        Returns
        -------
        type
            The paired ``sp.Function`` subclass.

        Raises
        ------
        ValueError
            If the option is unknown or has no sympy counterpart.

        Notes
        -----
        This is the entry point used by ``Genome.sympy_func`` /
        ``node.to_dict``, which carry the *name* of the aggregation and
        activation; ids and callables are accepted for symmetry with the
        gene layer.
        """
        if isinstance(func, (int, np.integer)):
            return self.get_sympy_func(func)
        if isinstance(func, str):
            if func not in self.name2index:
                raise ValueError(f"option {func!r} is not registered in {self!r}")
            return self.get_sympy_func(self.name2index[func])
        if callable(func):
            return self.get_sympy_func(self.name2index[self.name_of(func)])
        raise ValueError(f"{func!r} must be an option id, name or registered kernel")

    def sympy_module(self, backend: str = "numpy") -> dict:
        """Build the ``sp.lambdify`` module mapping for the sympy counterparts.

        Parameters
        ----------
        backend : str, optional
            ``"numpy"`` only -- the JAX backend disappeared with the evoxc
            rewrite, and sympy codegen is host-only.

        Returns
        -------
        dict
            ``{sympy_class_name: partial(numerical_eval, numpy)}`` for every
            counterpart *and* ``extra_sympy_classes`` entry that defines
            ``numerical_eval`` -- i.e. the classes a generated expression may
            still reference by name (``SympyClip``).

        Raises
        ------
        ValueError
            If ``backend`` is anything other than ``"numpy"``.

        Notes
        -----
        The ``numpy`` module is bound to the keyword ``backend``, not
        positionally: ``numerical_eval(val, min_val, max_val, backend=np)``
        means a positional partial would land in ``val`` and break every
        evaluation (the pre-rewrite ``partial(fn, jnp_or_np)`` did exactly that).

        Pass the result inside the ``modules=`` list of ``sp.lambdify``::

            sp.lambdify(..., modules=["numpy", AGG.sympy_module(), ACT.sympy_module()])
        """
        if backend != "numpy":
            raise ValueError(
                f"unsupported sympy backend {backend!r}: only 'numpy' remains "
                "after the evoxc rewrite"
            )
        module = {}
        for sympy_cls in self.sympy_fns + self.extra_sympy_classes:
            if sympy_cls is not None and hasattr(sympy_cls, "numerical_eval"):
                module[sympy_cls.__name__] = partial(
                    sympy_cls.numerical_eval, backend=np
                )
        return module

    def random_index(self, key: Any) -> Any:
        """Draw a uniformly random option id from a stateless evoxc key.

        Parameters
        ----------
        key : evoxc PRNG key
            A key obtained from ``evoxc.random.key(seed)`` / ``evoxc.random.split``.

        Returns
        -------
        ``() int32``
            Rank-0 option id uniform in ``[0, len(self))``; the ``-1`` identity
            sentinel is never drawn.

        Notes
        -----
        Graph code (this is the one evoxc op in this module): one
        ``evoxc.random.randint(key, (1,), 0, len(self))`` draw (``high`` is
        exclusive) reshaped to rank 0. The draw is requested as shape ``(1,)``
        rather than ``()`` because the numpy reference backend returns a bare
        NumPy scalar for a 0-d result and rejects it when it builds the output
        tensor; the reshape keeps the documented rank-0 contract without
        depending on that path. A caller sampling a *custom* option list of a
        different length calls the same two ops with ``len(options)``.
        """
        return evoxc.reshape(evoxc.random.randint(key, (1,), 0, len(self)), ())
