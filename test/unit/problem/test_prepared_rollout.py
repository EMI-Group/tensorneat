"""Unit tests for the prepared policy query of the RL rollouts (``_common.prepared_query``).

What this module proves
-----------------------
A rollout queries one network once per timestep, so the loop-invariant half of a
forward pass must be derived **once per generation**. Two pieces cooperate:

* :func:`._common.prepared_query` returns a genome-free callable unchanged and
  calls ``act_fn.prepare(batch)`` exactly once for a preparable one;
* the three shipped RL problems call it *before* their ``while_loop``, so one
  generation reaches ``_NetworkActFn.prepare`` exactly once even when the loop
  body runs ``max_step`` times.

Oracles used here
-----------------
The plain host callable itself (object identity for the unchanged case, and a
recording fake for "exactly once"), and the documented rollout contract — a
``CartPole(max_step=5)`` generation evaluated through ``problem.evaluate`` with a
counting ``_NetworkActFn`` subclass. The forward pass is a test policy, so the
assertion is about *how often* the pass is prepared, never about CartPole
physics: that is ``./test_cartpole.py``'s job.

Gotchas this module encodes
---------------------------
* ``_NetworkActFn.prepare`` is called **inside** the trace: the frozen record and
  the payload are graph values, so the counter is incremented while the graph is
  built, not while it runs. A regression back to one derivation per timestep
  therefore shows up as ``prepare_calls == max_step`` (five), not as a runtime
  cost.
* The fake payload only has to be batch-first: ``_prepared_query`` reads its
  static population size and the unchunked path never slices it.
* ``prepared_query`` decides on the presence of a ``prepare`` attribute, not on
  the concrete type, so a plain function must come back as the *same object*.

No ``jax``/``torch``, no network, no GPU, backend ``"numpy"``.
"""

from __future__ import annotations

from collections import namedtuple

import numpy as np

#: Population of the fake payload the counting rollout is evaluated on.
POP = 4

#: Horizon of the counting CartPole generation; also the number of queries a
#: per-timestep derivation would pay.
MAX_STEP = 5

#: Four batch-first symbolic leaves; only ``node_keys``' leading axis is read.
_FakePayload = namedtuple(
    "_FakePayload", ["node_keys", "node_attrs", "conn_keys", "conn_attrs"]
)


def test_prepared_query_returns_a_plain_callable_unchanged():
    """Invariant: a callable without ``prepare`` is returned as the same object.

    Oracle: object identity. The unit suite and user kernels pass plain
    ``enp``-only closures; wrapping one in a new object would hide it from
    ``evaluate`` and (for a closure over graph values) could change the trace.

    Assertion: ``prepared_query(act_fn, batch) is act_fn`` and the callable is
    never invoked by the helper.
    """
    from tensorneat.problem.rl._common import prepared_query

    calls = []

    def act_fn(batch):
        """A genome-free policy: the helper must not call it."""
        calls.append(batch)
        return batch

    assert prepared_query(act_fn, 12) is act_fn, (
        "a plain act_fn must come through unchanged; wrapping it would change what "
        "the problem queries"
    )
    assert calls == [], f"prepared_query must not query a plain callable, got {calls}"


def test_prepared_query_prepares_a_preparable_policy_once():
    """Invariant: a policy exposing ``prepare`` is asked for its pass exactly once.

    Oracle: the recorded call argument. The rollout hands ``prepared_query`` the
    static episode count ``B = P * R``; the prepared callable it returns is what
    the loop body queries, so ``prepare`` itself must not be reached again.

    Assertion: ``prepare`` was called once with ``batch`` and the helper returned
    the object ``prepare`` produced (not the act_fn and not a wrapper).
    """
    from tensorneat.problem.rl._common import prepared_query

    prepared = []

    class Preparable:
        """A policy that hoists its pass: the shape ``_NetworkActFn`` exposes."""

        def prepare(self, batch):
            """Record the batch and hand back the per-query callable."""
            prepared.append(batch)
            return "prepared-query"

        def __call__(self, batch):
            """A prepared pass must not be rebuilt through ``__call__``."""
            raise AssertionError("prepared_query returned the act_fn, not its query")

    act_fn = Preparable()

    assert prepared_query(act_fn, 7) == "prepared-query"
    assert prepared == [7], (
        f"prepare must be called exactly once with the batch size, got {prepared}"
    )


def test_cartpole_generation_prepares_the_policy_once(
    problem, common, state, fake_schedule, run_evaluate
):
    """Invariant: one ``CartPole(max_step=5)`` generation prepares the pass once, not per step.

    Oracle: a counting subclass of the real ``_NetworkActFn``. The rollout's
    ``max_step`` queries are all served by the single callable
    ``prepared_query`` returns; if ``_rollout`` went back to querying ``act_fn``
    directly, ``_NetworkActFn.__call__`` would prepare a fresh pass at every
    timestep and the counter would reach ``max_step``.

    Assertion: after one traced ``problem.evaluate`` the counter is exactly
    ``1`` (a per-timestep regression would make it ``MAX_STEP``), and the
    returned ``(POP,)`` fitness is finite — the test policy always pushes right
    and the default reset distribution keeps every episode alive for at least
    one step.

    Notes:
        ``source`` builds a batch-first payload *inside* the trace, because
        ``_NetworkActFn.prepare`` is reached while the graph is built; the fake
        schedule supplies ``POP`` as the problem's static population. The
        policy's forward ignores the schedule and returns ``(rows, 2)`` logits,
        which is all ``_force_from_logits`` reads.
    """
    import evoxc
    from evoxc import numpy as enp

    import tensorneat.host_payload as host_payload

    class CountingActFn(host_payload._NetworkActFn):
        """The production act_fn with a counter on the per-generation prepare."""

        def __init__(self, *args, **kwargs):
            """Set up the production object and the prepare counter."""
            super().__init__(*args, **kwargs)
            self.prepare_calls = 0

        def prepare(self, batch_rows):
            """Count this preparation and delegate to the real one."""
            self.prepare_calls += 1
            return super().prepare(batch_rows)

    def source():
        """Return the trace-time ``(state, transformed, payload)`` triple."""
        payload = _FakePayload(
            enp.zeros((POP, 2), dtype=evoxc.int32),
            enp.zeros((POP, 2, 1), dtype=evoxc.float32),
            enp.zeros((POP, 1, 2), dtype=evoxc.int32),
            enp.zeros((POP, 1, 1), dtype=evoxc.float32),
        )
        return state, None, payload

    def prepare(state_arg, transformed, payload, samples):
        """Stand-in for ``algorithm.prepare_forward``: record the batch shape."""
        return {"samples": samples, "rows": int(payload.node_keys.shape[0])}

    def forward(state_arg, transformed, prepared, inputs):
        """A constant push-right policy: logits ``(0, 1)`` for every row."""
        rows = int(inputs.shape[0])
        assert prepared["rows"] == POP, (
            "the prepared record must describe the problem's population, got "
            f"{prepared['rows']} for POP={POP}"
        )
        return enp.stack(
            [
                enp.zeros((rows,), dtype=evoxc.float32),
                enp.ones((rows,), dtype=evoxc.float32),
            ],
            axis=1,
        )

    act_fn = CountingActFn(source, forward, prepare)
    environment = problem.CartPole(max_step=MAX_STEP)
    schedule = fake_schedule(pop_size=POP)

    fitness = run_evaluate(environment, state, act_fn, schedule)

    assert act_fn.prepare_calls == 1, (
        "one generation must prepare the policy pass exactly once; a per-timestep "
        f"derivation would call prepare {MAX_STEP} times, got {act_fn.prepare_calls}"
    )
    assert fitness.shape == (POP,), (
        f"CartPole.evaluate must return ({POP},) fitness, got {fitness.shape}"
    )
    assert np.isfinite(np.asarray(fitness)).all(), (
        f"the constant push-right policy must return finite survival times, got {fitness}"
    )
