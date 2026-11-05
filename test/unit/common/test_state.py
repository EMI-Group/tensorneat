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
* a missing *entry* is one exception with two bases: attribute access raises
  ``common.state.StateKeyError``, which is both an ``AttributeError`` (so
  ``getattr(state, name, default)``, ``hasattr`` and ``copy.deepcopy`` read the
  miss as "absent") and a ``KeyError`` (so the legacy ``except KeyError`` sites
  keep working), while ``state["missing"]`` raises the plain builtin
  ``KeyError`` — the separation that keeps mapping code unchanged;
* ``copy.copy``/``copy.deepcopy``/``pickle`` round-trip the container: a copy
  keeps every entry in order, and a deep copy or an unpickle carries a fresh
  mutable payload instead of sharing one (``copy.deepcopy``'s
  ``getattr(x, "__deepcopy__", None)`` probe is the caller the dual base exists
  for);
* ``save``/``load`` round-trip through a real file in ``tmp_path``, and a broken
  file is reported, never absorbed into a default ``State``;
* the ``evoxc`` pytree registration (``tree_flatten``/``tree_unflatten``) keeps
  entry names in *aux data* and values as leaves, so a ``State`` can cross a
  graph boundary as a container.

Oracles: plain Python equality/identity, ``np.array_equal`` for the stored
tensor payloads (``State`` defines no ``__eq__``, so a copy is compared
structurally through :func:`_content`/:func:`_assert_same_content`, a literal expected
``repr``, the pre-call snapshot of the original container, the raw ``pickle``
payload read back through a recording stand-in class, and evoxc's own pytree
utilities. There is no numerical oracle here because ``State`` stores values
without interpreting them — that separation is itself the contract under test.

``tensorneat.common`` is reached through the ``common`` fixture (``./conftest.py``),
which resolves the package lazily; ``evoxc`` supplies the one ``evoxc`` RNG-key
payload and the pytree utilities the registration test uses. No ``jax``/``torch``,
no network, no GPU.
"""

from __future__ import annotations

import copy
import io
import pickle

import evoxc
import numpy as np
import pytest


@pytest.fixture
def state(common):
    """A two-entry ``State`` (``a=1``, ``b=[2, 3]``) that every mutator test starts from.

    ``b`` is a mutable value on purpose: "immutable container" means the *entry
    set* cannot change in place, and a test that stored only scalars could not
    tell a copied container from a shared one.
    """
    return common.State(a=1, b=[2, 3])


@pytest.fixture
def payload_state(common):
    """A ``State`` carrying one value of each kind a real generation holds.

    ``pop_size`` is an ordinary scalar, ``config`` a plain configuration dict,
    ``randkey`` an ``evoxc`` RNG key (an ``evoxc.Tensor``) and ``arr`` a
    ``(2, 3)`` float32 tensor. The four kinds are what the copy/deepcopy/pickle
    tests need: only a mutable payload can show a copy that shares state, and
    only a tensor payload shows that an unpickle really recreated it.
    """
    return common.State(
        pop_size=4,
        config={"activation": "tanh", "layers": [3, 4]},
        randkey=evoxc.random.key(0),
        arr=np.arange(6, dtype=np.float32).reshape(2, 3),
    )


def _content(state):
    """Record a container's structural content: ``(key names, name -> value)``.

    Args:
        state (State): The container to record.

    Returns:
        tuple: ``(tuple(state.registered_keys()), {name: state[name]})`` — the
        entry order and the stored values, reached through item access so the
        record does not depend on the attribute spelling it is checking.

    Notes:
        ``State`` defines no ``__eq__``, so this pair (compared by
        :func:`_assert_same_content`) is how "the same container" is asserted;
        a bare ``==`` would fall back to identity and only ever pass for ``is``.
    """
    names = tuple(state.registered_keys())
    return names, {name: state[name] for name in names}


def _value_equal(left, right):
    """Whether two stored values hold the same content without being the same object.

    Oracle: ``np.array_equal`` on the numpy payloads (a plain array, or an
    ``evoxc.Tensor`` RNG key read through its ``numpy()`` accessor, which is the
    only way to compare a key — it has no ``__eq__`` returning a bool), and plain
    ``==`` for the scalar and container kinds (a dict, a tuple, an int).
    """
    if isinstance(left, evoxc.Tensor) or isinstance(right, evoxc.Tensor):
        if not (isinstance(left, evoxc.Tensor) and isinstance(right, evoxc.Tensor)):
            return False
        return np.array_equal(left.numpy(), right.numpy())
    if isinstance(left, np.ndarray) or isinstance(right, np.ndarray):
        return np.array_equal(np.asarray(left), np.asarray(right))
    return left == right


def _assert_same_content(actual, expected):
    """Assert ``actual`` holds exactly ``expected``'s entries, in order, with equal values.

    Args:
        actual (State): The container produced by a copy/deepcopy/pickle.
        expected (State): The container it was derived from.

    Returns:
        tuple: The ``(names, values)`` record of ``actual``, for further
        assertions on identity.

    Notes:
        The oracle of every copying test: an entry dropped, reordered or
        silently replaced fails here even when the returned object has the right
        type, which is exactly the shape of a partial container.
    """
    assert type(actual) is type(expected)
    expected_names, expected_values = _content(expected)
    names, values = _content(actual)
    assert names == expected_names
    assert values.keys() == expected_values.keys()
    for name, value in values.items():
        assert _value_equal(value, expected_values[name]), f"entry {name!r} differs"
    return names, values


def _snapshot(state):
    """Record a container's observable content, to be compared after a mutator call.

    Args:
        state (State): The container to record.

    Returns:
        tuple: ``(repr(state), tuple of entry names, payload dict)``, where the
        payload is the documented pickle payload with list entries re-listed so
        that an in-place edit of a stored list is visible too.

    Notes:
        The before/after oracle of the mutator tests: a method that secretly
        edits the original container is caught even when it also returns a
        fresh object.
    """
    payload = state.__getstate__()
    values = {
        key: (list(value) if isinstance(value, list) else value)
        for key, value in payload.items()
    }
    return repr(state), tuple(state.registered_keys()), values


def _pickle_payload(common, raw):
    """Read the entry dict out of a saved ``State`` *without* using its accessors.

    Args:
        common: The ``tensorneat.common`` package (for ``common.State``).
        raw (bytes): The bytes of a file written by ``State.save``.

    Returns:
        dict: The mapping the pickle machinery hands to ``State.__setstate__``,
        i.e. the on-disk payload.

    Notes:
        A ``pickle.Unpickler`` that resolves ``tensorneat.common.state.State`` to
        a recording subclass captures the state dict while unpickling, so the
        assertion compares the *file* against a literal, never the reloaded
        container against itself.
    """
    captured = {}

    class _Recorder(common.State):
        def __setstate__(self, state):
            captured.update(state)

    class _Unpickler(pickle.Unpickler):
        def find_class(self, module, name):
            if module == common.State.__module__ and name == "State":
                return _Recorder
            return super().find_class(module, name)

    unpickled = _Unpickler(io.BytesIO(raw)).load()
    assert isinstance(unpickled, _Recorder), "the file did not hold a State"
    return captured


# --------------------------------------------------------------------------- #
# reading
# --------------------------------------------------------------------------- #


def test_empty_state_has_no_entries(common):
    """Invariant: ``State()`` is legal and holds nothing — empty key view, no membership.

    Oracle: ``list(state.registered_keys()) == []``, ``"a" not in state`` and the
    literal ``"State ({})"`` repr; an empty container is what a ``Pipeline``
    builds a fresh generation on, so it must not need a dummy entry.
    """
    empty = common.State()

    assert list(empty.registered_keys()) == []
    assert "a" not in empty
    assert repr(empty) == "State ({})"


def test_attribute_access_reads_an_entry(common, state):
    """Invariant: ``state.a`` returns the stored object itself, not a copy or a wrapper.

    Oracle: ``is`` identity against the very object handed to the constructor and
    ``==`` against a literal for the scalar, so a container that deep-copied on
    read (or returned a tensor wrapper) fails.
    """
    marker = [2, 3]
    built = common.State(a=1, b=marker)

    assert built.a == 1
    assert built.b is marker

    # The shared fixture reads the same way.
    assert state.a == 1
    assert state.b == [2, 3]


def test_missing_entry_is_an_attribute_error_and_a_key_error(common, state):
    """Invariant: ``state.typo`` raises ``StateKeyError``: an ``AttributeError`` and a ``KeyError``.

    Oracle: the documented dual base of ``common.state.StateKeyError`` and the
    concrete class ``State.__getattr__`` raises — one exception object that
    satisfies both bases, because ``getattr``/``hasattr``/``copy.deepcopy``
    swallow the ``AttributeError`` half while the legacy ``except KeyError``
    around ``state.pop_size`` catches the other. Item access is pinned as the
    *plain* builtin ``KeyError`` so mapping code keeps its usual semantics.
    """
    with pytest.raises(common.state.StateKeyError) as direct:
        state.typo  # the read itself is the assertion

    exc = direct.value
    assert type(exc) is common.state.StateKeyError
    assert isinstance(exc, AttributeError)
    assert isinstance(exc, KeyError)
    assert isinstance(exc, LookupError)  # KeyError's base, a free consequence
    assert "typo" in str(exc)

    # The ``AttributeError`` half: the probing spellings now read the miss as
    # "absent". ``hasattr`` is the documented cost of the dual base — a typo is
    # silent there and loud only at a bare read or an explicit ``except``.
    assert hasattr(state, "typo") is False
    assert getattr(state, "typo", None) is None

    # The other half deliberately does not leak into the mapping protocol:
    # ``state[...]`` is always the plain builtin ``KeyError``, never
    # ``StateKeyError`` and never an ``AttributeError``.
    with pytest.raises(KeyError) as item:
        state["typo"]
    assert type(item.value) is KeyError
    assert not isinstance(item.value, AttributeError)
    assert not isinstance(item.value, common.state.StateKeyError)
    assert "typo" in str(item.value)

    # Real entries are reachable through every spelling, so the guard is narrow.
    assert state.a == 1
    assert state.b == [2, 3]
    assert getattr(state, "a") == 1
    assert state["a"] == 1
    assert hasattr(state, "a") is True


def test_contains_tests_entry_names(state):
    """Invariant: ``"a" in state`` is an entry-name test, and a *value* is not an entry.

    Oracle: the documented membership semantics; ``1 in state`` must be ``False``
    even though ``state.a == 1``, otherwise ``in`` silently becomes a value scan.
    """
    assert "a" in state
    assert "b" in state
    assert "typo" not in state

    assert 1 not in state  # the value of ``a``, not an entry name
    assert "1" not in state


def test_registered_keys_preserve_insertion_order(common, state):
    """Invariant: ``registered_keys()`` yields the entries in insertion order.

    Oracle: the constructor's keyword order (``("a", "b")``), a reversed
    constructor call (``State(b=..., a=...)``) that proves the order is not
    sorted, and, after ``register(c=3, d=4)``, the appended order — the order is
    load-bearing because ``tree_flatten`` lays the leaves out with it.
    """
    assert tuple(state.registered_keys()) == ("a", "b")
    assert tuple(common.State(b=[2, 3], a=1).registered_keys()) == ("b", "a")

    extended = state.register(c=3, d=4)
    assert tuple(extended.registered_keys()) == ("a", "b", "c", "d")

    # The original keeps its own order and content.
    assert tuple(state.registered_keys()) == ("a", "b")


def test_repr_uses_the_legacy_format(common, state):
    """Invariant: ``repr(state) == "State ({'a': 1, 'b': [2, 3]})"``.

    Oracle: the documented legacy format kept for log and checkpoint
    comparability; the assertion is on the literal string, for the shared
    fixture, for the empty container and for a string-valued entry.
    """
    assert repr(state) == "State ({'a': 1, 'b': [2, 3]})"
    assert repr(common.State()) == "State ({})"
    assert repr(common.State(x="s")) == "State ({'x': 's'})"


# --------------------------------------------------------------------------- #
# mutators: immutability and direction of the error
# --------------------------------------------------------------------------- #


def test_register_returns_a_new_state_and_leaves_the_original_untouched(state):
    """Invariant: ``register(c=3)`` yields a container with ``a,b,c``; ``state`` still has ``a,b``.

    Oracle: identity (``new is not state``), the pre-call snapshot re-read after
    the call, and the refusal of in-place attribute assignment — the
    immutability contract every ``Pipeline`` step depends on.
    """
    before = _snapshot(state)

    new = state.register(c=3)

    assert new is not state
    assert type(new) is type(state)
    assert list(new.registered_keys()) == ["a", "b", "c"]
    assert new.a == 1
    assert new.b is state.b
    assert new.c == 3

    assert _snapshot(state) == before
    assert "c" not in state

    with pytest.raises(AttributeError, match="immutable"):
        state.a = 5
    assert state.a == 1


def test_register_rejects_an_existing_key(common, state):
    """Invariant: ``register`` over an existing entry raises ``ConfigError`` (it never overwrites).

    Oracle: the documented "use ``update`` for that"; the two operations must stay
    distinguishable, or a rename bug becomes a silent value change. The error is
    a :class:`TensorNEATError` that names the offending key, and the container is
    unchanged afterwards.
    """
    before = _snapshot(state)

    with pytest.raises(common.errors.ConfigError, match="already exists") as raised:
        state.register(a=9)

    assert isinstance(raised.value, common.errors.TensorNEATError)
    assert raised.value.message == str(raised.value)  # the hierarchy stores the message
    assert "a" in str(raised.value)
    assert _snapshot(state) == before


def test_update_replaces_values_immutably(state):
    """Invariant: ``update(a=9)`` replaces ``a`` and keeps ``b``; the original keeps ``a == 1``.

    Oracle: the resulting container's key set equals the original's (``update``
    never adds), identity of the two objects, and the pre-call snapshot of the
    original re-read afterwards.
    """
    before = _snapshot(state)
    original_b = state.b

    new = state.update(a=9)

    assert new is not state
    assert type(new) is type(state)
    assert list(new.registered_keys()) == ["a", "b"]
    assert new.a == 9
    assert new.b is original_b

    assert _snapshot(state) == before
    assert state.a == 1


def test_update_rejects_an_unknown_key(common, state):
    """Invariant: ``update`` of a missing entry raises ``ConfigError`` (it never creates).

    Oracle: the documented direction rule — ``update`` and ``register`` are
    deliberately not one operation — plus the untouched original.
    """
    before = _snapshot(state)

    with pytest.raises(common.errors.ConfigError, match="does not exist"):
        state.update(c=3)

    assert "c" not in state
    assert _snapshot(state) == before


def test_remove_drops_entries_and_keeps_the_original(state):
    """Invariant: ``remove("b")`` yields ``{a}``; the original still has ``a`` and ``b``.

    Oracle: the documented "returns a new container"; also covers ``remove()``
    with several names at once, which must drop exactly those, and the empty
    argument list, which drops nothing.
    """
    before = _snapshot(state)

    without_b = state.remove("b")

    assert without_b is not state
    assert list(without_b.registered_keys()) == ["a"]
    assert "b" not in without_b

    assert list(state.remove("a", "b").registered_keys()) == []

    keep_all = state.remove()
    assert keep_all is not state
    assert list(keep_all.registered_keys()) == ["a", "b"]

    assert _snapshot(state) == before


def test_remove_rejects_an_unknown_key(common, state):
    """Invariant: ``remove`` of a missing entry raises ``ConfigError``.

    Oracle: the documented "so a typo cannot silently remove nothing" — the
    failure mode this rule exists for; the original is untouched after it.
    """
    before = _snapshot(state)

    with pytest.raises(common.errors.ConfigError, match="does not exist"):
        state.remove("typo")

    assert _snapshot(state) == before


# --------------------------------------------------------------------------- #
# checkpointing
# --------------------------------------------------------------------------- #


def test_save_and_load_round_trip_in_tmp_path(common, state, tmp_path):
    """Invariant: ``State.load(path)`` reconstructs every entry with equal values.

    Oracle: equality of the key tuple and of each value against the object that
    was saved, ``type(loaded) is State`` (``load`` is a classmethod), no shared
    mutable sub-object, and the *raw pickle payload* read back by hand — which
    must equal the literal ``{"a": 1, "b": [2, 3]}``. ``tmp_path`` keeps the
    file out of the repository.
    """
    path = tmp_path / "state.pkl"

    state.save(path)

    assert path.is_file()
    raw = path.read_bytes()
    assert raw  # a real, non-empty checkpoint

    assert _pickle_payload(common, raw) == {"a": 1, "b": [2, 3]}

    loaded = common.State.load(path)

    assert type(loaded) is common.State
    assert loaded is not state
    assert tuple(loaded.registered_keys()) == tuple(state.registered_keys())
    assert list(loaded.registered_keys()) == ["a", "b"]
    assert loaded.a == state.a
    assert loaded.b == state.b
    assert loaded.b is not state.b  # unpickling copies: no shared state
    assert repr(loaded) == repr(state)

    # ``save``/``load`` accept a string path too (``os.PathLike`` is documented).
    as_str = tmp_path / "state_str.pkl"
    state.save(str(as_str))
    assert common.State.load(str(as_str)).b == [2, 3]


def test_load_propagates_a_missing_or_corrupt_file_error(common, tmp_path):
    """Invariant: a missing file raises ``OSError``, a corrupt one ``pickle.UnpicklingError``.

    Oracle: the documented "never swallowed into a default ``State``" — a truncated
    checkpoint must fail loudly, because a default container would resume a run
    from nowhere with plausible numbers. Both a garbage file and a truncated
    real checkpoint are tried.
    """
    missing = tmp_path / "does_not_exist.pkl"
    assert not missing.exists()

    with pytest.raises(OSError):
        common.State.load(missing)
    with pytest.raises(FileNotFoundError):  # the concrete OSError subclass
        common.State.load(missing)

    garbage = tmp_path / "garbage.pkl"
    garbage.write_bytes(b"not a pickle\x00\x01\x02")
    with pytest.raises(pickle.UnpicklingError):
        common.State.load(garbage)

    truncated = tmp_path / "truncated.pkl"
    complete = tmp_path / "complete.pkl"
    common.State(a=1, b=[2, 3]).save(complete)
    truncated.write_bytes(complete.read_bytes()[:16])
    with pytest.raises(pickle.UnpicklingError):
        common.State.load(truncated)


# --------------------------------------------------------------------------- #
# copying: copy/copy.deepcopy (the AttributeError half's reason to exist)
# --------------------------------------------------------------------------- #


def test_copy_keeps_every_entry_and_shares_the_payload(common, payload_state):
    """Invariant: ``copy.copy`` yields a distinct ``State`` with the same entries and shared values.

    Oracle: :func:`_content` against the original, entry by entry, plus ``is``
    identity on the mutable payload — a shallow copy sharing the stored objects
    is what it is *for*, and the assertions that follow (the container is a new
    object, the order is insertion order) keep it from being a no-op.
    """
    shallow = copy.copy(payload_state)

    assert type(shallow) is common.State
    assert shallow is not payload_state
    names, values = _assert_same_content(shallow, payload_state)
    assert names == ("pop_size", "config", "randkey", "arr")  # insertion order, not sorted

    # A shallow copy shares the values it copied.
    assert values["arr"] is payload_state.arr
    assert values["randkey"] is payload_state.randkey
    assert values["config"] is payload_state.config
    assert values["pop_size"] == 4

    # A shallow copy of an empty container stays empty; nothing invents an entry.
    assert _content(copy.copy(common.State())) == ((), {})


def test_deepcopy_keeps_every_entry_with_a_fresh_payload(common, payload_state):
    """Invariant: ``copy.deepcopy`` reproduces all entries but shares no mutable payload.

    Oracle: :func:`_content` field by field, ``is not`` on the numpy array and the
    config dict, and a write into the copy's array that must not reach the
    original — no shared mutable state. This is also the caller the
    ``AttributeError`` half of ``StateKeyError`` exists for: ``copy.deepcopy``
    probes ``getattr(x, "__deepcopy__", None)`` on the *instance*, so a bare
    ``KeyError`` there aborted the copy instead of producing a container.
    """
    deep = copy.deepcopy(payload_state)

    assert type(deep) is common.State
    assert deep is not payload_state
    names, values = _assert_same_content(deep, payload_state)
    assert names == ("pop_size", "config", "randkey", "arr")  # nothing dropped

    assert values["arr"] is not payload_state.arr
    assert np.array_equal(values["arr"], payload_state.arr)
    assert values["config"] is not payload_state.config
    assert values["config"] == payload_state.config
    assert values["randkey"] is not payload_state.randkey
    assert np.array_equal(values["randkey"].numpy(), payload_state.randkey.numpy())
    assert values["pop_size"] == 4

    # No shared mutable state: editing the copy's array leaves the original alone,
    # and editing the copy's dict does not reach the original's dict either.
    deep.arr[0, 0] = -1.0
    deep.config["layers"].append(5)
    assert payload_state.arr[0, 0] == 0.0
    assert payload_state.config == {"activation": "tanh", "layers": [3, 4]}

    # The empty container has no payload to copy and must not gain one.
    assert _content(copy.deepcopy(common.State())) == ((), {})


def test_pickle_round_trip_restores_a_usable_state(common, payload_state, tmp_path):
    """Invariant: ``pickle.loads(pickle.dumps(state))`` is a usable ``State`` with the same content.

    Oracle: :func:`_content` field by field (arrays compared with
    ``np.array_equal``, the RNG key through its ``numpy()`` payload), ``type(...)
    is common.State``, a fresh tensor payload, reads through *both* accessors,
    and a second round trip through :meth:`State.save`/:meth:`State.load` in
    ``tmp_path`` — a container that answered only one spelling would be a broken
    checkpoint.
    """
    restored = pickle.loads(pickle.dumps(payload_state))

    assert type(restored) is common.State
    assert restored is not payload_state
    names, values = _assert_same_content(restored, payload_state)
    assert names == ("pop_size", "config", "randkey", "arr")  # order survives
    assert values["arr"] is not payload_state.arr  # unpickling copies the payload
    assert np.array_equal(values["arr"], payload_state.arr)
    assert type(values["randkey"]) is evoxc.Tensor
    assert values["config"] == {"activation": "tanh", "layers": [3, 4]}

    # Usable straight away: every entry reads through both spellings.
    assert restored.pop_size == 4
    assert restored["pop_size"] == 4
    assert restored.config == {"activation": "tanh", "layers": [3, 4]}
    assert restored["config"]["layers"] == [3, 4]
    assert tuple(restored["arr"].shape) == (2, 3)
    assert "arr" in restored
    assert hasattr(restored, "randkey") is True
    with pytest.raises(common.state.StateKeyError):
        restored.typo

    # Still a checkpointable State: save + load once more through a real file.
    path = tmp_path / "payload.pkl"
    restored.save(path)
    reloaded = common.State.load(path)
    assert type(reloaded) is common.State
    _assert_same_content(reloaded, payload_state)

    # And a real one: the mutators still work on the restored container.
    extended = reloaded.register(extra=7)
    assert extended.extra == 7
    assert "extra" not in reloaded
    assert tuple(reloaded.registered_keys()) == tuple(payload_state.registered_keys())


# --------------------------------------------------------------------------- #
# pytree plumbing (a State crosses a graph boundary as a container of leaves)
# --------------------------------------------------------------------------- #


def test_tree_flatten_puts_values_in_children_and_names_in_aux_data(common, state):
    """Invariant: ``tree_flatten() == ([1, [2, 3]], ["a", "b"])``.

    Oracle: the literal pair; the names travel as *aux data* so that a kernel
    which rebuilds the leaves cannot lose the key→value association. A reversed
    constructor call pins names and values to insertion order (not sorted), and
    the stored value is carried through unchanged (identity for the mutable one).
    """
    children, aux_data = state.tree_flatten()

    assert isinstance(children, list)
    assert isinstance(aux_data, list)
    assert children == [1, [2, 3]]
    assert aux_data == ["a", "b"]
    assert children[1] is state.b

    # Names and values stay paired positionally, in insertion order (not sorted).
    assert common.State(b=[2, 3], a=1).tree_flatten() == ([[2, 3], 1], ["b", "a"])

    # An empty container flattens to two empty halves, not to a sentinel entry.
    assert state.remove("a", "b").tree_flatten() == ([], [])


def test_tree_unflatten_round_trips_and_validates_lengths(common, state):
    """Invariant: unflatten(flatten(s)) rebuilds ``s``; a mismatched length raises ``ValueError``.

    Oracle: equality with the original container for the round-trip, and the
    documented ``ValueError`` for hand-built children — the two halves of the
    pytree contract. Tuples are accepted (only the length and the order matter).
    """
    children, aux_data = state.tree_flatten()

    rebuilt = common.State.tree_unflatten(aux_data, children)

    assert type(rebuilt) is common.State
    assert rebuilt is not state
    assert list(rebuilt.registered_keys()) == ["a", "b"]
    assert rebuilt.a == 1
    assert rebuilt.b == [2, 3]

    # Positional pairing, not key lookup: swapping the values swaps the entries.
    swapped = common.State.tree_unflatten(("a", "b"), ([2, 3], 1))
    assert swapped.a == [2, 3]
    assert swapped.b == 1

    # Tuples work as well as lists.
    assert common.State.tree_unflatten(("x",), (5,)).x == 5

    with pytest.raises(ValueError, match="entry names"):
        common.State.tree_unflatten(["a", "b"], [1])
    with pytest.raises(ValueError, match="entry names"):
        common.State.tree_unflatten(["a"], [1, 2])


def test_state_is_a_registered_evoxc_pytree_node(common, state):
    """Invariant: ``evoxc.tree_leaves`` lists the values and ``tree_map`` rebuilds a ``State``.

    Oracle: ``evoxc``'s own pytree utilities (the registration is done at import
    time in ``state.py``), which is the only observable consequence of
    ``register_pytree_node``; ``tree_map(lambda v: v, state)`` must come back as a
    ``State`` with the same entries, not as a tuple, and a leaf-wise map must be
    reflected in the rebuilt container under the *same* names.
    """
    import evoxc

    leaves, treespec = evoxc.tree_flatten(state)
    assert leaves == [1, 2, 3]  # the list entry is itself a pytree node
    assert treespec.type is common.State

    identity = evoxc.tree_map(lambda value: value, state)
    assert type(identity) is common.State
    assert list(identity.registered_keys()) == ["a", "b"]
    assert identity.b == [2, 3]

    mapped = evoxc.tree_map(lambda value: value + 1, common.State(x=1, y=2))
    assert type(mapped) is common.State
    assert list(mapped.registered_keys()) == ["x", "y"]
    assert (mapped.x, mapped.y) == (2, 3)

    # Rebuilding from the leaves alone restores the names: they live in aux data.
    rebuilt = evoxc.tree_unflatten([9, 8, 7], treespec)
    assert type(rebuilt) is common.State
    assert list(rebuilt.registered_keys()) == ["a", "b"]
    assert rebuilt.a == 9
    assert rebuilt.b == [8, 7]
