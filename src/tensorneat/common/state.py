"""Immutable host container for configuration, evoxc RNG keys and tensors.

Dependency DAG position: bottom of ``tensorneat.common`` (with
:mod:`~tensorneat.common.errors`); imported by every other node of the
package, imports none of them. Host-only module: no ``jax``/``torch``, no
NumPy's global random module, and no evoxc op — a ``State`` never participates in a
traced kernel except as a pytree that is flattened *across* a graph boundary.

Rationale: a generation is a pure function of ``(seed, config, shape)``.
``State`` carries the evoxc RNG keys explicitly so that no hidden global
randomness exists, and it is immutable so that a ``Pipeline`` step cannot
mutate a caller's copy: every mutator returns a **new** ``State``.

Naming rule: ``State`` keys are plain strings (``"nodes"``, ``"conns"``,
``"randkey"``, ...); tensors stored in a ``State`` keep the package-wide
batch-first convention (population on axis 0, static shapes, ``NaN`` = absent
float, ``I_INF`` = absent int index).

Error rule: a missing entry raises :class:`StateKeyError` from attribute access
(an ``AttributeError`` for ``getattr``/``hasattr``/``copy``/``pickle`` that is
also a ``KeyError`` for the legacy catch sites) and the plain ``KeyError`` from
``state["missing"]``.
"""

import pickle

import evoxc

from .errors import ConfigError


class StateKeyError(AttributeError, KeyError):
    """A missing ``State`` entry, reported to two kinds of caller at once.

    The dual base is deliberate, and each half has a concrete caller:

    * ``AttributeError`` is what the object protocols around an instance's
      namespace require: :func:`copy.deepcopy` probes ``__deepcopy__`` on the
      instance with :func:`getattr` and reads a missing name (i.e. an
      ``AttributeError``) as "no hook", :func:`copy.copy` and :mod:`pickle`
      take the same answer from ``__copy__``/``__reduce_ex__``/``__setstate__``,
      and the builtins :func:`getattr` (with a default) and :func:`hasattr`
      swallow exactly this type. Only an ``AttributeError`` is absorbed, so any
      other type escapes out of ``copy.deepcopy`` and out of
      ``getattr(state, "pop", None)`` -- the probe
      ``Pipeline._shape_signature`` uses to reach its own "call setup() first"
      ``TensorNEATError``.
    * ``KeyError`` is what the legacy catch sites expect -- in particular
      ``Genome._read_pop_size`` in ``genome/base.py``, which wraps
      ``state.pop_size`` in ``except KeyError`` and raises its own
      ``TensorNEATError``; that file belongs to a sibling node and may not be
      edited, so the exception *class* (not the catch site) is what keeps both
      behaviours working without a cross-node change.

    Cost: ``hasattr(state, "typo")`` now answers ``False`` instead of
    propagating the failure, so a typo is loud only at a bare ``state.typo``
    read, at an explicit ``except KeyError``/``except AttributeError``, or
    through :meth:`State.__getitem__` (always a plain ``KeyError``).

    Args:
        message (str): The message, naming the missing entry.
    """


class State:
    """Immutable host container over a ``dict`` of named values.

    A ``State`` behaves like a frozen namespace: attribute access reads a key,
    while ``register`` / ``update`` / ``remove`` return a *new* ``State`` and
    leave the original untouched. It may hold anything a kernel-level
    ``State`` needs — the genome/algorithm/problem configuration objects, the
    evoxc RNG keys, and the batch-first tensors of the current generation.

    Args:
        **kwargs: The initial entries. Values are arbitrary Python objects;
            dict/tuple values are not flattened unless ``evoxc``'s pytree
            machinery is applied to the ``State`` itself.

    Raises:
        tensorneat.common.errors.ConfigError: If it is ever necessary to
            validate a stored value (e.g. a tensor whose population axis does
            not match the configured ``pop_size``), that validation raises
            here — never inside a graph.

    Notes:
        Registered as an ``evoxc`` pytree node at import time
        (``evoxc.register_pytree_node(State, State.tree_flatten,
        State.tree_unflatten)``), so a ``State`` can be passed into and out of
        an ``@evoxc.defn`` graph as a container of leaves rather than as an
        opaque host object.
    """

    def __init__(self, **kwargs):
        """Wrap the keyword arguments as the initial entry set.

        Args:
            **kwargs: The initial entries; insertion order is preserved by
                :meth:`registered_keys`, :meth:`tree_flatten` and
                :meth:`__repr__`.
        """
        # The entries live in ``__dict__`` under one reserved name so that
        # ``__getattr__`` (which is only consulted for names that normal lookup
        # misses) can reach them without recursing into itself.
        self.__dict__["state_dict"] = kwargs

    def registered_keys(self):
        """Names of the entries currently held.

        Returns:
            A ``dict_keys`` view over the entry names, in insertion order. It
            is a live view of an immutable mapping, so callers may iterate it
            but must not rely on modifying the ``State`` through it.
        """
        return self.state_dict.keys()

    def __getattr__(self, name):
        """Read an entry by name.

        Args:
            name (str): Entry name.

        Returns:
            The stored value, unchanged (config object, evoxc key, tensor, ...).

        Raises:
            StateKeyError: If ``name`` is not a registered entry. It is both an
                ``AttributeError`` -- so ``hasattr``, ``getattr(state, name,
                default)``, ``copy.deepcopy`` and the pickle path treat a
                missing entry as "absent" instead of crashing -- and a
                ``KeyError``, so the legacy ``except KeyError`` around
                ``state.pop_size`` in ``genome/base.py`` keeps raising its own
                error; see :class:`StateKeyError` for the full rationale.
                Consequence: ``hasattr`` answers ``False`` for a typo instead
                of propagating the failure.

        Notes:
            ``__getattr__`` is consulted only for names that normal lookup
            misses, so the real attributes and methods (``state_dict``,
            ``registered_keys``, ...) never come through here.
        """
        # ``self.__dict__`` is a type-level descriptor, so it never reaches
        # ``__getattr__``; ``.get`` also keeps an unpickling half-built object
        # (no ``state_dict`` yet) from recursing instead of failing.
        state_dict = self.__dict__.get("state_dict")
        if state_dict is None or name not in state_dict:
            raise StateKeyError(f"State has no entry {name!r}")
        return state_dict[name]

    def __getitem__(self, item):
        """Read an entry by name through the mapping protocol (``state["pop"]``).

        Args:
            item (str): Entry name.

        Returns:
            The stored value, unchanged -- the same object ``state.<item>`` gives.

        Raises:
            KeyError: If ``item`` is not a registered entry. Always the plain
                builtin, never :class:`StateKeyError`: mapping code that catches
                ``KeyError`` and only that keeps its usual semantics, and the
                message names the missing entry.
        """
        state_dict = self.__dict__.get("state_dict")
        if state_dict is None or item not in state_dict:
            raise KeyError(f"State has no entry {item!r}")
        return state_dict[item]

    def __contains__(self, item):
        """Membership test over entry names (``"nodes" in state``).

        Args:
            item (str): Entry name.

        Returns:
            bool: ``True`` if the entry exists.
        """
        return item in self.state_dict

    def __repr__(self):
        """Debug representation listing every entry name and value.

        Returns:
            str: ``"State ({'a': 1, ...})"`` — the legacy format, kept so logs
            and saved checkpoints read the same.
        """
        return f"State ({self.state_dict})"

    def __setattr__(self, name, value):
        """Refuse attribute assignment: a ``State`` is immutable.

        Args:
            name (str): Attribute name.
            value: Ignored.

        Raises:
            AttributeError: Always. The entries are only ever changed by the
                mutators, which build a new ``State``; the internal dict is
                reached through ``self.__dict__`` (and by ``__setstate__``),
                not through this hook.
        """
        raise AttributeError("State is immutable")

    def __getstate__(self):
        """Pickle support: hand the entry dict to :mod:`pickle`.

        Returns:
            dict: A shallow copy of the entries.
        """
        return self.state_dict.copy()

    def __setstate__(self, state):
        """Pickle support: restore the entry dict without ``__setattr__``.

        Args:
            state (dict): The entries produced by :meth:`__getstate__`.
        """
        self.__dict__["state_dict"] = state

    def register(self, **kwargs):
        """Return a new ``State`` with extra entries.

        Args:
            **kwargs: Entries to add.

        Returns:
            State: A new container holding the old entries plus ``kwargs``.

        Raises:
            tensorneat.common.errors.ConfigError: If any key already exists —
                ``register`` never overwrites; use :meth:`update` for that.
        """
        for key in kwargs:
            if key in self.state_dict:
                raise ConfigError(f"Key {key} already exists in state")
        return State(**{**self.state_dict, **kwargs})

    def update(self, **kwargs):
        """Return a new ``State`` with existing entries replaced.

        Args:
            **kwargs: Entries to overwrite.

        Returns:
            State: A new container with the same keys and the replaced values.

        Raises:
            tensorneat.common.errors.ConfigError: If any key is missing —
                ``update`` never creates entries; use :meth:`register` for
                that.
        """
        for key in kwargs:
            if key not in self.state_dict:
                raise ConfigError(f"Key {key} does not exist in state")
        return State(**{**self.state_dict, **kwargs})

    def remove(self, *keys):
        """Return a new ``State`` without the named entries.

        Args:
            *keys (str): Entry names to drop.

        Returns:
            State: A new container without those entries.

        Raises:
            tensorneat.common.errors.ConfigError: If any key is missing, so a
                typo cannot silently remove nothing.
        """
        for key in keys:
            if key not in self.state_dict:
                raise ConfigError(f"Key {key} does not exist in state")
        return State(**{k: v for k, v in self.state_dict.items() if k not in keys})

    def save(self, file_name):
        """Pickle the ``State`` to ``file_name`` (checkpointing).

        Args:
            file_name (str | os.PathLike): Destination path; written in binary
                mode, overwriting any existing file.

        Returns:
            None

        Raises:
            OSError: Propagated unchanged from ``open`` (missing directory,
                permissions).

        Notes:
            Host-only. Pickling requires every stored value to be picklable;
            compiled executables are not, so the ``Pipeline`` keeps those out
            of the ``State``.
        """
        with open(file_name, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, file_name):
        """Restore a ``State`` previously written by :meth:`save`.

        Args:
            file_name (str | os.PathLike): Source path.

        Returns:
            State: The unpickled container (a new object, no shared state with
            the saving process).

        Raises:
            OSError: Propagated unchanged from ``open``.
            pickle.UnpicklingError: Propagated unchanged for a corrupt file;
                never swallowed into a default ``State``.
        """
        with open(file_name, "rb") as f:
            return pickle.load(f)

    def tree_flatten(self):
        """Flatten into evoxc pytree ``(children, aux_data)``.

        Returns:
            tuple: ``(children, aux_data)`` where ``children`` is the list of
            stored values in insertion order and ``aux_data`` is the matching
            list of entry names. The names must travel as aux data (not as
            leaves) so that unflattening restores the key→value association
            even when a kernel rebuilds the leaves.
        """
        return list(self.state_dict.values()), list(self.state_dict.keys())

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        """Rebuild a ``State`` from :meth:`tree_flatten` output.

        Args:
            aux_data: The entry names produced by :meth:`tree_flatten`.
            children: The leaf values, positionally matched to ``aux_data``.

        Returns:
            State: A new container.

        Raises:
            ValueError: If ``len(aux_data) != len(children)`` (a pytree
                misuse, e.g. hand-built children).
        """
        names = list(aux_data)
        values = list(children)
        if len(names) != len(values):
            raise ValueError(
                "State.tree_unflatten: "
                f"{len(names)} entry names but {len(values)} leaf values"
            )
        return cls(**dict(zip(names, values)))


# Register with evoxc's pytree registry so a `State` can cross a graph boundary
# as a container of leaves. Host code that never traces is unaffected.
evoxc.register_pytree_node(State, State.tree_flatten, State.tree_unflatten)
