"""Tests for scooter company biddings."""
import numpy as np
import pytest

from bayesian_scooters.strategies import (
    AggressiveScooterCompany,
    Namer,
    NeutralScooterCompany,
    ScooterCompany,
    UniformScooterCompany,
)


def test_get_word_list() -> None:
    """Assert that a list of strings is returned."""
    names = Namer().get_word_list()
    assert type(names) == list
    for n in names:
        assert type(n) == str


def test_get_name() -> None:
    """Assert that a string is returned."""
    name = Namer().get_random_name()
    assert type(name) == str


@pytest.mark.parametrize(
    "company, N",
    [
        (UniformScooterCompany(), 10),
        (NeutralScooterCompany(), 5),
        (AggressiveScooterCompany(), 7),
        (UniformScooterCompany(), None),
        (NeutralScooterCompany(), None),
        (AggressiveScooterCompany(), None),
    ],
)
def test_bid(company: ScooterCompany, N: int) -> None:
    """Assert that the bid has the proper rows and columns."""
    bid = company.bid(N=N)
    if N is not None:
        assert bid.shape[0] == N
    assert "company" in bid.columns
    assert type(bid.company.values[0]) == str
    assert "units" in bid.columns
    assert type(bid.units.values[0]) == np.int64
    assert "consideration" in bid.columns
    assert type(bid.consideration.values[0]) == np.float64
    assert "priority" in bid.columns
    assert type(bid.priority.values[0]) == np.int64
