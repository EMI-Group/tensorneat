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
"""

import evoxc


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
        raise NotImplementedError(
            "store kwargs in an internal dict without exposing mutators"
        )

    def registered_keys(self):
        """Names of the entries currently held.

        Returns:
            A ``dict_keys`` view over the entry names, in insertion order. It
            is a live view of an immutable mapping, so callers may iterate it
            but must not rely on modifying the ``State`` through it.
        """
        raise NotImplementedError(
            "return the live key view of the internal dict"
        )

    def __getattr__(self, name):
        """Read an entry by name.

        Args:
            name (str): Entry name.

        Returns:
            The stored value, unchanged (config object, evoxc key, tensor, ...).

        Raises:
            KeyError: If ``name`` is not a registered entry. Deliberately a
                ``KeyError`` and not an ``AttributeError``: a missing *entry*
                is a configuration/typo bug, and catching it as
                ``AttributeError`` would make it silent (``hasattr`` would
                swallow it).
        """
        raise NotImplementedError(
            "return the stored value for name or raise KeyError"
        )

    def __contains__(self, item):
        """Membership test over entry names (``"nodes" in state``).

        Args:
            item (str): Entry name.

        Returns:
            bool: ``True`` if the entry exists.
        """
        raise NotImplementedError(
            "test membership of an entry name"
        )

    def __repr__(self):
        """Debug representation listing every entry name and value.

        Returns:
            str: ``"State ({'a': 1, ...})"`` — the legacy format, kept so logs
            and saved checkpoints read the same.
        """
        raise NotImplementedError(
            "render the entry dict in the legacy State (...) format"
        )

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
        raise NotImplementedError(
            "return a new State with the given entries added, rejecting duplicates"
        )

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
        raise NotImplementedError(
            "return a new State with the given entries replaced, rejecting unknown keys"
        )

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
        raise NotImplementedError(
            "return a new State without the given entries, rejecting unknown keys"
        )

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
        raise NotImplementedError(
            "pickle the state to file_name"
        )

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
        raise NotImplementedError(
            "unpickle a state from file_name"
        )

    def tree_flatten(self):
        """Flatten into evoxc pytree ``(children, aux_data)``.

        Returns:
            tuple: ``(children, aux_data)`` where ``children`` is the list of
            stored values in insertion order and ``aux_data`` is the matching
            list of entry names. The names must travel as aux data (not as
            leaves) so that unflattening restores the key→value association
            even when a kernel rebuilds the leaves.
        """
        raise NotImplementedError(
            "return the stored values as children and their names as aux data"
        )

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
        raise NotImplementedError(
            "rebuild a State by zipping names with children"
        )


# Register with evoxc's pytree registry so a `State` can cross a graph boundary
# as a container of leaves. Host code that never traces is unaffected.
evoxc.register_pytree_node(State, State.tree_flatten, State.tree_unflatten)
