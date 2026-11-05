"""Regression tests for the case matrix's declared-target flag (``cases.py``).

A case whose initial population already beats its ``fitness_target`` cannot report a
meaningful generations-to-target count, so it declares ``gens_to_target=False`` and the worker
then measures with ``target=None``. ``cartpole_p100`` is that case: its fitness is surviving
steps with a 200-step rollout horizon, and the initial population already reaches 145/200/200 on
seeds 42/43/44, so the recorded metric read 2/1/1 and carried no information.
The flag must survive into ``as_dict()`` because the case-matrix snapshot is what lands in a run
record's ``case_config`` — a flag that only lives on the dataclass would be lost in the evidence.
"""

from __future__ import annotations

import pytest

from benchmarks import cases


def test_cartpole_declares_its_target_uninformative() -> None:
    """The recorded expectation: ``cases.get_cases(["cartpole_p100"])[0].gens_to_target is False``."""
    assert cases.get_cases(["cartpole_p100"])[0].gens_to_target is False


def test_the_declared_flag_lands_in_the_case_matrix_snapshot() -> None:
    """``as_dict()`` carries ``gens_to_target`` as a JSON boolean, not a missing key."""
    snapshot = cases.get_cases(["cartpole_p100"])[0].as_dict()
    assert "gens_to_target" in snapshot
    assert snapshot["gens_to_target"] is False


def test_no_other_case_suppresses_its_target() -> None:
    """Every other case keeps the default (``True``), so its cell keeps a generation count."""
    flags = {case.name: case.as_dict()["gens_to_target"] for case in cases.get_cases()}
    assert flags == {
        "xor_p100": True,
        "xor_p500": True,
        "xor_p1000": True,
        "xor3d_p100": True,
        "xor3d_p500": True,
        "xor3d_p1000": True,
        "custom_pagie_p500": True,
        "cartpole_p100": False,
    }


def test_suppressing_the_metric_keeps_the_documented_solved_threshold() -> None:
    """The flag suppresses the *recorded metric*, not the case's fitness target or horizon."""
    cartpole = cases.get_cases(["cartpole_p100"])[0]
    assert cartpole.fitness_target == 180.0
    assert cartpole.problem_config["max_step"] == 200
    assert cartpole.impls == ("new",), "no legacy equivalent, so the old side is not planned"


def test_the_flag_defaults_to_an_informative_target() -> None:
    """A case constructed without the flag keeps the target (the dataclass default is ``True``)."""
    case = cases.Case(
        name="probe",
        family="func_fit",
        impls=("new",),
        pop_size=8,
        generation_limit=1,
        fitness_target=-0.5,
        algorithm_config={"class": "NEAT"},
        genome_config={"class": "DefaultGenome"},
        problem_config={"class": "XOR"},
    )
    assert case.gens_to_target is True
    assert case.as_dict()["gens_to_target"] is True


@pytest.mark.parametrize("requested", ["cartpole_p100", "cartpole_p100,cartpole_p100"])
def test_selecting_cartpole_returns_that_case_once(requested: str) -> None:
    """Selection is by name and drops a duplicate, so one request is one case."""
    selected = cases.get_cases([name for name in requested.split(",")])
    assert [case.name for case in selected] == ["cartpole_p100"]
