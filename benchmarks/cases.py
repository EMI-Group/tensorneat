"""Benchmark case matrix: the logical workload both implementations are measured on.

A :class:`Case` declares the problem, the genome and the NEAT configuration as
plain data (the ``*_config`` dicts), plus the budget, the fitness target and the
seed. :meth:`Case.build` turns those dicts into live objects for one
implementation, and :meth:`Case.as_dict` turns the *same* dicts into the
JSON-safe record the reporter renders — so a number in ``RESULTS.md`` cannot
drift from the object that produced it.

This module is stdlib-only at import time: ``numpy``/``jax`` are imported lazily
inside the builders that need them, so ``python benchmarks/run_benchmark.py
--list`` works on a bare interpreter that has neither installed.
"""

from __future__ import annotations

import copy
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from typing import Any

__all__ = ["Case", "CASES", "all_names", "get_cases"]

_FAMILIES = frozenset({"func_fit", "rl"})
"""Valid values of :attr:`Case.family`; they group the rows of the case matrix."""

_IMPLS = frozenset({"old", "new"})
"""Valid values of an entry of :attr:`Case.impls`."""

_XOR_MS_PER_GENERATION: dict[int, float] = {100: 51.0, 500: 109.0, 1000: 157.0}
"""Declared legacy CPU baseline (root ``CONTEXT.md``, 1-thread-bound box).

These are a sanity check printed next to the *measured* old row; the verdict is
computed from the row measured in the same run on the same box.
"""

_XOR3D_FITNESS_TARGET: float = -0.05
"""``-mse`` target of the XOR3d cases: mse <= 0.05, i.e. clearly beyond a linear fit.

The target must lie below the trivial linear-baseline fitness of ``-0.25``,
otherwise generations-to-target is satisfied from generation 1 (a purely linear
3->1 network already fits 3-bit parity with mse 0.25). The former ``-0.5`` was
met by every initial population, so the report's "gens to target" column read 1
for every xor3d run and carried no information.
"""


def _pagie_as_jax(point: Any) -> Any:
    """Pagie polynomial ``1/(1+x**-4) + 1/(1+y**-4)`` as a traced ``(1,)`` array.

    The legacy ``CustomFuncFit`` evaluates the function with ``jax.vmap`` and
    also calls it eagerly on ``low_bounds``, so this returns a ``jax.numpy``
    array and accepts both a numpy array and a tracer (``jax.numpy`` handles
    numpy input fine).

    Args:
        point: A 1-D point of shape ``(2,)``.

    Returns:
        A ``(1,)`` ``float32`` jax array holding the Pagie value.
    """
    import jax.numpy as jnp

    x, y = point[0], point[1]
    res = 1.0 / (1.0 + x**-4) + 1.0 / (1.0 + y**-4)
    return jnp.array([res], dtype=jnp.float32)


def _pagie_as_numpy(point: Any) -> Any:
    """Pagie polynomial ``1/(1+x**-4) + 1/(1+y**-4)`` as a plain ``(1,)`` array.

    The evoxc rewrite builds the dataset host-side with
    ``np.apply_along_axis``, so plain numpy is what it wants.

    Args:
        point: A 1-D point of shape ``(2,)``.

    Returns:
        A ``(1,)`` ``float32`` numpy array holding the Pagie value.
    """
    import numpy as np

    x, y = point[0], point[1]
    res = 1.0 / (1.0 + x**-4) + 1.0 / (1.0 + y**-4)
    return np.array([res], dtype=np.float32)


_FUNC_FACTORIES: dict[str, dict[str, Callable]] = {
    "pagie": {"old": _pagie_as_jax, "new": _pagie_as_numpy},
}
"""``problem_config["func"]`` -> per-implementation target-function builder.

The value named in a config is a builder-side factory key, not a JSON value: the
two implementations need different array types from the same formula.
"""


@dataclass(frozen=True, kw_only=True)
class Case:
    """One benchmark workload, declared as data.

    ``kw_only`` keeps the declaration order identical to the documented field
    order: ``seed``/``notes`` have defaults and sit before the required config
    dicts, which a positional dataclass rejects.

    Attributes:
        name: Case id, also the key in :data:`CASES` (e.g. ``"xor_p100"``).
        family: ``"func_fit"`` or ``"rl"``; groups the case in the report.
        impls: Implementations that can run it, ``("old", "new")`` or
            ``("new",)``.
        pop_size: Population size ``P``; axis 0 of every population tensor.
        generation_limit: Informational generation budget of a full run.
        fitness_target: Best fitness that counts as solved.
        seed: Base seed; repeat ``r`` of a run uses ``seed + r``.
        notes: Free-form remark shown in the case matrix.
        algorithm_config: NEAT keyword arguments plus a ``"class"`` key.
        genome_config: Genome keyword arguments plus a ``"class"`` key.
        problem_config: Problem keyword arguments plus a ``"class"`` key.
        reference_ms_per_generation: Declared legacy baseline keyed by population
            size; only the XOR cases carry measured numbers.
    """

    name: str
    family: str
    impls: tuple[str, ...]
    pop_size: int
    generation_limit: int
    fitness_target: float
    seed: int = 42
    notes: str = ""
    algorithm_config: dict[str, Any]
    genome_config: dict[str, Any]
    problem_config: dict[str, Any]
    # A factory, not a ``{}`` literal: a shared mutable dataclass default would
    # be rejected at class creation and shared between cases if it were not.
    reference_ms_per_generation: dict[int, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Reject a case that could only fail later, deep inside a run.

        Raises:
            ValueError: If the name is empty, the family or an implementation is
                unknown, or a config dict has no ``"class"`` key.
        """
        if not self.name:
            raise ValueError("case name must be a non-empty string")
        if self.family not in _FAMILIES:
            raise ValueError(
                f"case {self.name!r}: family must be one of {sorted(_FAMILIES)}, "
                f"got {self.family!r}"
            )
        if not self.impls or any(impl not in _IMPLS for impl in self.impls):
            raise ValueError(
                f"case {self.name!r}: impls must be a non-empty subset of "
                f"{sorted(_IMPLS)}, got {self.impls!r}"
            )
        for attr in ("algorithm_config", "genome_config", "problem_config"):
            cfg = getattr(self, attr)
            if not isinstance(cfg.get("class"), str):
                raise ValueError(
                    f"case {self.name!r}: {attr} needs a \"class\" key, got {cfg!r}"
                )

    # ------------------------------------------------------------------ build

    def build(self, ns: Any, impl: str) -> tuple[object, object]:
        """Construct the live ``(algorithm, problem)`` pair of one implementation.

        Every argument comes from the config dicts, so the JSON record rendered
        from :meth:`as_dict` and the objects that were actually measured cannot
        disagree.

        Args:
            ns: The imported implementation package: ``tensorneat`` for
                ``"new"``, ``tensorneat_legacy`` for ``"old"``.
            impl: ``"old"`` or ``"new"``; must be listed in :attr:`impls`.

        Returns:
            ``(algorithm, problem)``, ready for ``Pipeline(algorithm, problem)``.

        Raises:
            ValueError: If ``impl`` cannot run this case or a config names a
                class / function the package does not provide.
        """
        if impl not in self.impls:
            raise ValueError(
                f"case {self.name!r} does not run on impl {impl!r}; "
                f"available: {list(self.impls)}"
            )
        genome = self._build_genome(ns)
        algorithm = self._build_algorithm(ns, genome)
        problem = self._build_problem(ns, impl)
        return algorithm, problem

    def _build_genome(self, ns: Any) -> Any:
        """Build the genome described by :attr:`genome_config`."""
        cls, kwargs = _resolve(
            ns.genome, self.genome_config, f"case {self.name!r} genome"
        )
        transform = kwargs.pop("output_transform", None)
        if transform is not None:
            # ``None`` means "keep the class default" and is therefore *not*
            # passed: a transform owns the read-out, and CartPole must keep its
            # own argmax so the decision stays a traced branch.
            kwargs["output_transform"] = (
                getattr(ns.common.ACT, transform)
                if isinstance(transform, str)
                else transform
            )
        return cls(**kwargs)

    def _build_algorithm(self, ns: Any, genome: Any) -> Any:
        """Build the algorithm described by :attr:`algorithm_config`."""
        cls, kwargs = _resolve(
            ns.algorithm, self.algorithm_config, f"case {self.name!r} algorithm"
        )
        return cls(**kwargs, genome=genome)

    def _build_problem(self, ns: Any, impl: str) -> Any:
        """Build the problem described by :attr:`problem_config`."""
        cls, kwargs = _resolve(
            ns.problem, self.problem_config, f"case {self.name!r} problem"
        )
        if "func" in kwargs:
            return self._build_custom_func_fit(cls, kwargs, impl)
        return cls(**kwargs)

    def _build_custom_func_fit(self, cls: Any, kwargs: dict[str, Any], impl: str) -> Any:
        """Build a ``CustomFuncFit`` problem, resolving its ``"func"`` factory.

        The per-implementation divergence is deliberate, and it is the only one
        in this file: the legacy class vmaps the function under ``jax`` (so the
        function must return a traced array) and draws its samples from the
        *global* numpy RNG (so the process RNG is seeded immediately before
        construction), while the evoxc rewrite builds the dataset host-side
        with plain numpy and takes an explicit ``seed``. Same distribution,
        different draws — trajectories are not compared (see
        ``../CONTEXT.md``, "What Is Not Compared").
        """
        kwargs = dict(kwargs)
        func_name = kwargs.pop("func")
        if func_name not in _FUNC_FACTORIES:
            raise ValueError(
                f"case {self.name!r}: unknown func factory {func_name!r}; "
                f"known: {sorted(_FUNC_FACTORIES)}"
            )
        func = _FUNC_FACTORIES[func_name][impl]
        if impl == "old":
            import numpy as np

            np.random.seed(self.seed)
            return cls(func=func, **kwargs)
        return cls(func=func, seed=self.seed, **kwargs)

    # --------------------------------------------------------------- reporting

    def as_dict(self) -> dict:
        """Return a JSON-safe snapshot of the case.

        It is what the case-matrix table is rendered from and what lands in a
        ``RunResult``'s ``case_config`` field; the config dicts are deep-copied
        so a consumer cannot mutate the registered case through the record.

        Returns:
            A plain ``dict`` with string keys only.
        """
        return {
            "name": self.name,
            "family": self.family,
            "impls": list(self.impls),
            "pop_size": self.pop_size,
            "generation_limit": self.generation_limit,
            "fitness_target": self.fitness_target,
            "seed": self.seed,
            "notes": self.notes,
            "algorithm": self.algorithm_config["class"],
            "genome": self.genome_config["class"],
            "problem": self.problem_config["class"],
            "algorithm_config": copy.deepcopy(self.algorithm_config),
            "genome_config": copy.deepcopy(self.genome_config),
            "problem_config": copy.deepcopy(self.problem_config),
            "reference_ms_per_generation": {
                str(pop): ms for pop, ms in self.reference_ms_per_generation.items()
            },
        }


def _resolve(module: Any, cfg: dict[str, Any], what: str) -> tuple[Any, dict[str, Any]]:
    """Split a config dict into ``(class, kwargs)``.

    Args:
        module: The package namespace the class lives in (``ns.genome``,
            ``ns.algorithm`` or ``ns.problem``).
        cfg: The config dict; its ``"class"`` entry is removed from the kwargs.
        what: Human-readable label of the config, used in the error message.

    Returns:
        ``(class, kwargs)`` with ``"class"`` popped from ``kwargs``.

    Raises:
        ValueError: If ``module`` does not expose the named class.
    """
    kwargs = dict(cfg)
    class_name = kwargs.pop("class")
    try:
        return getattr(module, class_name), kwargs
    except AttributeError:
        raise ValueError(
            f"{what}: {getattr(module, '__name__', module)!r} has no class "
            f"{class_name!r}"
        ) from None


def _xor_family_case(
    *,
    name: str,
    pop_size: int,
    problem_class: str,
    num_inputs: int,
    fitness_target: float,
    reference_ms_per_generation: dict[int, float],
) -> Case:
    """Build one XOR / XOR3d case: sigmoid read-out, 20 species slots.

    The six XOR / XOR3d cases share this configuration and differ only in the
    problem, the input width, the population, the target and the reference
    baseline, so it is written once instead of six times.
    """
    return Case(
        name=name,
        family="func_fit",
        impls=("old", "new"),
        pop_size=pop_size,
        generation_limit=150,
        fitness_target=fitness_target,
        algorithm_config={
            "class": "NEAT",
            "pop_size": pop_size,
            "species_size": 20,
            "survival_threshold": 0.01,
        },
        genome_config={
            "class": "DefaultGenome",
            "num_inputs": num_inputs,
            "num_outputs": 1,
            "max_nodes": 50,
            "max_conns": 100,
            "output_transform": "sigmoid",
        },
        problem_config={"class": problem_class},
        reference_ms_per_generation=reference_ms_per_generation,
    )


_CASES_IN_ORDER: tuple[Case, ...] = (
    _xor_family_case(
        name="xor_p100",
        pop_size=100,
        problem_class="XOR",
        num_inputs=2,
        fitness_target=-0.1,
        reference_ms_per_generation=dict(_XOR_MS_PER_GENERATION),
    ),
    _xor_family_case(
        name="xor_p500",
        pop_size=500,
        problem_class="XOR",
        num_inputs=2,
        fitness_target=-0.1,
        reference_ms_per_generation=dict(_XOR_MS_PER_GENERATION),
    ),
    _xor_family_case(
        name="xor_p1000",
        pop_size=1000,
        problem_class="XOR",
        num_inputs=2,
        fitness_target=-0.1,
        reference_ms_per_generation=dict(_XOR_MS_PER_GENERATION),
    ),
    _xor_family_case(
        name="xor3d_p100",
        pop_size=100,
        problem_class="XOR3d",
        num_inputs=3,
        fitness_target=_XOR3D_FITNESS_TARGET,
        reference_ms_per_generation={},
    ),
    _xor_family_case(
        name="xor3d_p500",
        pop_size=500,
        problem_class="XOR3d",
        num_inputs=3,
        fitness_target=_XOR3D_FITNESS_TARGET,
        reference_ms_per_generation={},
    ),
    _xor_family_case(
        name="xor3d_p1000",
        pop_size=1000,
        problem_class="XOR3d",
        num_inputs=3,
        fitness_target=_XOR3D_FITNESS_TARGET,
        reference_ms_per_generation={},
    ),
    Case(
        name="custom_pagie_p500",
        family="func_fit",
        impls=("old", "new"),
        pop_size=500,
        generation_limit=150,
        fitness_target=-0.05,
        notes=(
            "per-impl divergence: the legacy side seeds the global numpy RNG "
            "and vmaps a jax-typed func; the new side samples from "
            "default_rng(seed) with a numpy func — same distribution, different draws"
        ),
        algorithm_config={
            "class": "NEAT",
            "pop_size": 500,
            "species_size": 20,
            "survival_threshold": 0.01,
        },
        genome_config={
            "class": "DefaultGenome",
            "num_inputs": 2,
            "num_outputs": 1,
            "max_nodes": 50,
            "max_conns": 100,
            "output_transform": "sigmoid",
        },
        problem_config={
            "class": "CustomFuncFit",
            "method": "sample",
            "num_samples": 500,
            "low_bounds": [-1.0, -1.0],
            "upper_bounds": [1.0, 1.0],
            "func": "pagie",
        },
    ),
    Case(
        name="cartpole_p100",
        family="rl",
        impls=("new",),
        pop_size=100,
        generation_limit=150,
        fitness_target=180.0,
        notes="new-only: the legacy library has no equivalent problem",
        algorithm_config={
            "class": "NEAT",
            "pop_size": 100,
            "species_size": 20,
            "survival_threshold": 0.01,
        },
        # No ``output_transform``: CartPole owns the argmax decision, so the
        # genome keeps its default and the branch stays traceable.
        genome_config={
            "class": "DefaultGenome",
            "num_inputs": 4,
            "num_outputs": 2,
            "max_nodes": 50,
            "max_conns": 100,
        },
        problem_config={"class": "CartPole", "max_step": 200, "repeat_times": 1},
    ),
)
"""The case matrix, in declaration order."""


def _by_name(cases: Iterable[Case]) -> dict[str, Case]:
    """Index ``cases`` by name, rejecting a duplicate.

    Args:
        cases: Cases in declaration order.

    Returns:
        ``{case.name: case}`` preserving insertion order.

    Raises:
        ValueError: If two cases share a name (the second would silently win).
    """
    registry: dict[str, Case] = {}
    for case in cases:
        if case.name in registry:
            raise ValueError(f"duplicate case name {case.name!r}")
        registry[case.name] = case
    return registry


CASES: dict[str, Case] = _by_name(_CASES_IN_ORDER)
"""Registered cases keyed by name; iteration order is declaration order."""


def all_names() -> list[str]:
    """Return every case name, in declaration order."""
    return list(CASES)


def get_cases(names: list[str] | None = None) -> list[Case]:
    """Resolve case names to :class:`Case` objects.

    Args:
        names: ``None`` (or ``["all"]``) means every case. Otherwise the named
            cases, in the requested order with duplicates dropped.

    Returns:
        The selected cases.

    Raises:
        ValueError: If a name is not in :data:`CASES`; the message lists every
            valid name.
    """
    if names is None or list(names) == ["all"]:
        return list(CASES.values())
    selected: list[Case] = []
    seen: set[str] = set()
    unknown: list[str] = []
    for name in names:
        case = CASES.get(name)
        if case is None:
            unknown.append(name)
        elif name not in seen:
            seen.add(name)
            selected.append(case)
    if unknown:
        raise ValueError(
            f"unknown case(s) {unknown!r}; valid names: {all_names()} "
            f"(or 'all' for the whole matrix)"
        )
    return selected
