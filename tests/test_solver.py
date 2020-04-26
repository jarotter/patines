"""Tests for solver module."""
from typing import Generic, TypeVar

import numpy as np
import pandas as pd
import pytest

from bayesian_scooters.solver import Contest, wasserstein_to_uniform
from bayesian_scooters.strategies import CustomScooterCompany

NUMBER_AGGRESIVE = 2
NUMBER_NEUTRAL = 1
NUMBER_CUSTOM = 1

T = TypeVar("T")


class Fixture(Generic[T]):
    """Pytest Fixture."""

    pass


@pytest.fixture
def contest() -> Contest:
    """Build contest to use in tests."""
    params = {"units": [35], "consideration": [1.5]}
    custom = CustomScooterCompany(params)
    return Contest(nagg=NUMBER_AGGRESIVE, nneu=NUMBER_NEUTRAL, custom=custom)


def test_null_proposals(contest: Fixture) -> None:
    """Check if null proposals where added.

    If (and only if) all were added, each company will have
    a minimum of 0 units. The sum acrosss all companies must
    be zero.

    Args:
        contest: pytest fixture for building the contest.
    """
    proposals = pd.concat(contest.cleaner.add_null_proposals(contest.proposals))
    test_value = proposals.groupby("company").agg({"units": min}).sum().values[0]
    assert test_value == 0


def test_scenario(contest: Fixture) -> None:
    """Test wether scenarios where built.

    If properly built, each company should be in the
    same number of scenarios.

    Args:
        contest: pytest fixture for building the contest.
    """
    scenarios = contest.cleaner.create_scenarios(contest.proposals)
    assert scenarios.groupby(["company"]).size().unique().size == 1


def test_wasserstein_to_uniform() -> None:
    """Test EMD calculation.

    It should be zero for the uniform distribution
    and positive elsewhere.
    """
    n = np.random.randint(1, 10)
    uniform_n = np.ones(n) / n
    assert np.isclose(wasserstein_to_uniform(uniform_n), 0)
    some_other_distribution = np.random.randn(size=n)
    some_other_distribution /= some_other_distribution.sum()
    assert wasserstein_to_uniform(some_other_distribution) > 0


def test_filter_number_of_units(contest: Fixture) -> None:
    """Test no scenario has more than 70 units.

    Args:
        contest: pytest fixture for building the contest.
    """
    scenarios = pd.DataFrame(
        {"scenario_id": [1, 1, 1, 2, 2], "units": [35, 20, 20, 30, 35]}
    )
    clean = contest.optimizer.filter_number_units(scenarios)
    assert clean.scenario_id.unique().values[0] == 2


def test_optimize_consideration(contest: Fixture) -> None:
    """Test that only the highest paying scenarios are kept.

    Args:
        contest: pytest fixture for building the contest.
    """
    scenarios = pd.DataFrame(
        {
            "scenario_id": [1, 1, 1, 2, 2, 3],
            "total_consideration": [100, 200, 300, 10, 20, 600],
        }
    )
    clean = contest.optimizer.optimize_consideration(scenarios).scenario_id.unique()
    assert 1 in clean
    assert 3 in clean
    assert clean.size == 2


def test_break_tie_priority(contest: Fixture) -> None:
    """Test that only scenarios with maximum number ones are kept.

    Args:
        contest: pytest fixture for building the contest.
    """
    scenarios = pd.DataFrame(
        {
            "scenario_id": [1, 1, 2, 2, 2, 3, 3, 3],
            "priority": [1, 1, 1, 2, 1, 3, 4, 4],
        }
    )
    clean = contest.optimizer.break_tie_priority(scenarios).scenario_id.unique()
    assert 1 in clean
    assert 2 in clean
    assert clean.size == 2


def test_break_tie_entropy(contest: Fixture) -> None:
    """Test that only most entropic distribution is kept.

    Args:
        contest: pytest fixture for building the contest.
    """
    scenarios = pd.DataFrame(
        {
            "scenario_id": [1, 1, 1, 2, 2],
            "units": [1, 1, 1, 1, 7],
        }
    )
    clean = contest.optimizer.break_tie_entropy(scenarios).scenario_id.unique()
    assert 1 in clean
    assert clean.size == 1


def test_optimize(contest: Fixture) -> None:
    """Assert only one answer is returned.

    Args:
        contest: pytest fixture for building the contest.
    """
    scenarios = contest.cleaner.create_scenarios(contest.proposals)
    optimal = contest.optimizer.optimize(scenarios)
    assert optimal.shape[0] == 0
    assert "company" in optimal.columns
    assert "units" in optimal.columns
    assert "consideration" in optimal.columns


def test_create_participants(contest: Fixture) -> None:
    """Assert participants are properly created.

    Args:
        contest: pytest fixture for building the contest.
    """
    assert (
        len(contest.participants) == NUMBER_AGGRESIVE + NUMBER_NEUTRAL + NUMBER_CUSTOM
    )


def test_recieve_proposals(contest: Fixture) -> None:
    """Assert proposals are correctly stored.

    Both the listed and joint proposals should include every company.

    Args:
        contest: pytest fixture for building the contest.
    """
    assert len(contest.proposals) == NUMBER_AGGRESIVE + NUMBER_NEUTRAL + NUMBER_CUSTOM
    total_rows = 0
    for p in contest.proposals:
        assert type(p) == pd.DataFrame
        total_rows += p.shape[0]
    assert type(contest.joint_proposals) == pd.DataFrame
    assert contest.joint_proposals.shape[0] == total_rows


def test_winner(contest: Fixture) -> None:
    """Assert the optimizer's answer has proper form.

    Though most is directly inheriited from test_optimize

    Args:
        contest: pytest fixture for building the contest.
    """
    winner = contest.get_winners()
    assert winner.shape[0] == 0
