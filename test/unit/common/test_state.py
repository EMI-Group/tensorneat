"""Unit tests for ``tensorneat.common.state`` — the immutable host container.

What this module proves
-----------------------
``State`` is the one host object every other node of the package threads through
a generation: it carries the genome/algorithm/problem configuration, the evoxc
RNG keys and the batch-first tensors, and it must be **immutable** so that a
``Pipeline`` step cannot mutate a caller's copy. So the tests are about
algebra and observability, not arithmetic:

* ``register``/``update``/``remove`` each return a *new* container and leave the
  original untouched (checked by identity of the returned object and by
  re-reading the original after the call);
* each mutator refuses the wrong direction — ``register`` a duplicate,
  ``update``/``remove`` a missing key — with
  :class:`tensorneat.common.errors.ConfigError`, so a typo cannot silently do
  nothing;
* a missing *entry* raises ``KeyError`` (deliberately not ``AttributeError``:
  ``hasattr`` would swallow that, and a missing entry is a configuration bug);
* ``save``/``load`` round-trip through a real file in ``tmp_path``, and a broken
  file is reported, never absorbed into a default ``State``;
* the ``evoxc`` pytree registration (``tree_flatten``/``tree_unflatten``) keeps
  entry names in *aux data* and values as leaves, so a ``State`` can cross a
  graph boundary as a container.

Oracles: plain Python equality/identity and the documented string format. There
is no numerical oracle here because ``State`` stores values without interpreting
them — that separation is itself the contract under test.

``tensorneat.common`` is reached through the ``common`` fixture (``./conftest.py``)
because ``import tensorneat`` still raises from the ``src/`` design skeletons.
No ``jax``/``torch``, no network.
"""

from __future__ import annotations

import pytest


@pytest.fixture
def state(common):
    """A two-entry ``State`` (``a=1``, ``b=[2, 3]``) that every mutator test starts from.

    ``b`` is a mutable value on purpose: "immutable container" means the *entry
    set* cannot change in place, and a test that stored only scalars could not
    tell a copied container from a shared one.
    """
    return common.State(a=1, b=[2, 3])


# --------------------------------------------------------------------------- #
# reading
# --------------------------------------------------------------------------- #


def test_empty_state_has_no_entries(common):
    """Invariant: ``State()`` is legal and holds nothing — empty key view, no membership.

    Oracle: ``list(state.registered_keys()) == []`` and ``"a" not in state``; an
    empty container is what a ``Pipeline`` builds a fresh generation on, so it
    must not need a dummy entry.
    """
    pytest.skip(
        "not implemented: State() empty key view and membership "
        "(owner: common/state.State.__init__)"
    )


def test_attribute_access_reads_an_entry(state):
    """Invariant: ``state.a`` returns the stored object itself, not a copy or a wrapper.

    Oracle: ``is`` identity for a mutable value and ``==`` for a scalar, so a
    container that deep-copied on read (or returned a tensor wrapper) fails.
    """
    pytest.skip(
        "not implemented: attribute read returns the stored value unchanged "
        "(owner: common/state.State.__getattr__)"
    )


def test_missing_entry_raises_key_error_and_escapes_hasattr(state):
    """Invariant: an unknown name raises ``KeyError``, and ``hasattr`` propagates it.

    Oracle: the documented error *type* — not ``AttributeError``, because
    ``hasattr(state, "typo")`` swallowing the failure is exactly the silent bug the
    choice of ``KeyError`` prevents. Asserted both directly and through
    ``hasattr``.
    """
    pytest.skip(
        "not implemented: KeyError for a missing entry, propagated through hasattr "
        "(owner: common/state.State.__getattr__)"
    )


def test_contains_tests_entry_names(state):
    """Invariant: ``"a" in state`` is an entry-name test, and a *value* is not an entry.

    Oracle: the documented membership semantics; ``1 in state`` must be ``False``
    even though ``state.a == 1``, otherwise ``in`` silently becomes a value scan.
    """
    pytest.skip(
        "not implemented: __contains__ tests names, not values "
        "(owner: common/state.State.__contains__)"
    )


def test_registered_keys_preserve_insertion_order(state):
    """Invariant: ``registered_keys()`` yields the entries in insertion order.

    Oracle: the constructor's keyword order (``("a", "b")``) and, after
    ``register(c=3)``, the appended order — the order is load-bearing because
    :meth:`tree_flatten` uses it to lay out the leaves.
    """
    pytest.skip(
        "not implemented: registered_keys insertion order, including after register "
        "(owner: common/state.State.registered_keys)"
    )


def test_repr_uses_the_legacy_format(state):
    """Invariant: ``repr(state) == "State ({'a': 1, 'b': [2, 3]})"``.

    Oracle: the documented legacy format kept for log and checkpoint
    comparability; the assertion is on the literal string.
    """
    pytest.skip(
        "not implemented: repr == 'State ({...})' legacy format "
        "(owner: common/state.State.__repr__)"
    )


# --------------------------------------------------------------------------- #
# mutators: immutability and direction of the error
# --------------------------------------------------------------------------- #


def test_register_returns_a_new_state_and_leaves_the_original_untouched(state):
    """Invariant: ``register(c=3)`` yields a container with ``a,b,c``; ``state`` still has ``a,b``.

    Oracle: identity (``new is not state``) plus re-reading the original after the
    call — the immutability contract every ``Pipeline`` step depends on.
    """
    pytest.skip(
        "not implemented: register returns a new State and the original is unchanged "
        "(owner: common/state.State.register)"
    )


def test_register_rejects_an_existing_key(state):
    """Invariant: ``register`` over an existing entry raises ``ConfigError`` (it never overwrites).

    Oracle: the documented "use ``update`` for that"; the two operations must stay
    distinguishable, or a rename bug becomes a silent value change.
    """
    pytest.skip(
        "not implemented: ConfigError for register on an existing key "
        "(owner: common/state.State.register)"
    )


def test_update_replaces_values_immutably(state):
    """Invariant: ``update(a=9)`` replaces ``a`` and keeps ``b``; the original keeps ``a == 1``.

    Oracle: the resulting container's key set equals the original's (``update``
    never adds) and identity of the two objects.
    """
    pytest.skip(
        "not implemented: update replaces named entries only and leaves the original intact "
        "(owner: common/state.State.update)"
    )


def test_update_rejects_an_unknown_key(state):
    """Invariant: ``update`` of a missing entry raises ``ConfigError`` (it never creates).

    Oracle: the documented direction rule — ``update`` and ``register`` are
    deliberately not one operation.
    """
    pytest.skip(
        "not implemented: ConfigError for update on a missing key "
        "(owner: common/state.State.update)"
    )


def test_remove_drops_entries_and_keeps_the_original(state):
    """Invariant: ``remove("b")`` yields ``{a}``; the original still has ``a`` and ``b``.

    Oracle: the documented "returns a new container"; also covers ``remove()``
    with several names at once, which must drop exactly those.
    """
    pytest.skip(
        "not implemented: remove drops the named entries and leaves the original intact "
        "(owner: common/state.State.remove)"
    )


def test_remove_rejects_an_unknown_key(state):
    """Invariant: ``remove`` of a missing entry raises ``ConfigError``.

    Oracle: the documented "so a typo cannot silently remove nothing" — the
    failure mode this rule exists for.
    """
    pytest.skip(
        "not implemented: ConfigError for remove on a missing key "
        "(owner: common/state.State.remove)"
    )


# --------------------------------------------------------------------------- #
# checkpointing
# --------------------------------------------------------------------------- #


def test_save_and_load_round_trip_in_tmp_path(common, state, tmp_path):
    """Invariant: ``State.load(path)`` reconstructs every entry with equal values.

    Oracle: equality of the key tuple and of each value against the object that
    was saved (and ``isinstance`` of the rebuilt container, since ``load`` is a
    classmethod). ``tmp_path`` keeps the file out of the repository.
    """
    pytest.skip(
        "not implemented: save/load round-trip of all entries through tmp_path "
        "(owner: common/state.State.save, common/state.State.load)"
    )


def test_load_propagates_a_missing_or_corrupt_file_error(common, tmp_path):
    """Invariant: a missing file raises ``OSError``, a corrupt one ``pickle.UnpicklingError``.

    Oracle: the documented "never swallowed into a default ``State``" — a truncated
    checkpoint must fail loudly, because a default container would resume a run
    from nowhere with plausible numbers.
    """
    pytest.skip(
        "not implemented: OSError for a missing checkpoint and UnpicklingError for a corrupt one "
        "(owner: common/state.State.load)"
    )


# --------------------------------------------------------------------------- #
# pytree plumbing (a State crosses a graph boundary as a container of leaves)
# --------------------------------------------------------------------------- #


def test_tree_flatten_puts_values_in_children_and_names_in_aux_data(state):
    """Invariant: ``tree_flatten() == ([1, [2, 3]], ["a", "b"])``.

    Oracle: the literal pair; the names travel as *aux data* so that a kernel
    which rebuilds the leaves cannot lose the key→value association.
    """
    pytest.skip(
        "not implemented: tree_flatten children/aux_data split "
        "(owner: common/state.State.tree_flatten)"
    )


def test_tree_unflatten_round_trips_and_validates_lengths(common, state):
    """Invariant: unflatten(flatten(s)) rebuilds ``s``; a mismatched length raises ``ValueError``.

    Oracle: equality with the original container for the round-trip, and the
    documented ``ValueError`` for hand-built children — the two halves of the
    pytree contract.
    """
    pytest.skip(
        "not implemented: tree_unflatten round-trip and ValueError on a length mismatch "
        "(owner: common/state.State.tree_unflatten)"
    )


def test_state_is_a_registered_evoxc_pytree_node(state):
    """Invariant: ``evoxc.tree_leaves`` lists the values and ``tree_map`` rebuilds a ``State``.

    Oracle: ``evoxc``'s own pytree utilities (the registration is done at import
    time in ``state.py``), which is the only observable consequence of
    ``register_pytree_node``; ``tree_map(lambda v: v, state)`` must come back as a
    ``State`` with the same entries, not as a tuple.
    """
    pytest.skip(
        "not implemented: evoxc.tree_leaves/tree_map see a State as named leaves "
        "(owner: common/state.py pytree registration)"
    )
