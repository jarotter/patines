"""Required classes for running a contest."""
from itertools import product
from typing import List

import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance

from bayesian_scooters.strategies import (
    AggressiveScooterCompany,
    NeutralScooterCompany,
    ScooterCompany,
)


class ProposalCleaner:
    """Utilities for cleaning generated proposals."""

    def __init__(self) -> None:
        pass

    def add_null_proposals(self, proposals: List[pd.DataFrame]) -> List[pd.DataFrame]:
        """Add the null 0 units proposal to each company.

        Args:
            proposals: List of DataFrames with each company's proposal

        Returns:
            A list of proposals in which every company now has a proposal with
                0 units as last priority.
        """
        new_proposals = []

        for p in proposals:
            N = p.shape[0]
            name = p.company.values[0]
            pp = p.append(
                {"company": name, "units": 0, "consideration": 0, "priority": N + 1},
                ignore_index=True,
            ).reset_index(drop=True)
            new_proposals.append(pp)
        return new_proposals

    def create_scenarios(self, proposals: List[pd.DataFrame]) -> pd.DataFrame:
        """Create combinations of companies.

        Cartesian product of every proposal, across
        all companies, creating a long dataframe in which
        every combination is uniquely identified.

        Args:
            proposals: List of DataFrames containing each company's proposals.

        Returns:
            DataFrame in which every row is a possible scenario combining the
                given proposals.
        """
        participants = [p.company.values[0] for p in proposals]
        proposals = self.add_null_proposals(proposals)
        priority_prod = (
            pd.DataFrame(
                product(*(p.priority for p in proposals)), columns=participants
            )
            .reset_index(drop=False)
            .rename(columns={"index": "scenario_id"})
            .melt(id_vars="scenario_id", var_name="company", value_name="priority")
            .merge(pd.concat(proposals), how="outer", on=["company", "priority"])
        )
        return priority_prod


def wasserstein_to_uniform(p: np.array) -> float:
    """Compute Wasserstein's metric to discrete uniform distribution.

    Args:
        p: Probability distribution. If not normalized, it will be.

    Returns:
        Wasserstein distance to the discrete uniform distribution.
    """
    p /= p.sum()
    n = p.size
    uniform = np.ones(n) / n
    return wasserstein_distance(p, uniform)


class ContestOptimizer:
    """Utilities to run determine the winner of a contest."""

    def __init__(self) -> None:
        pass

    def filter_number_units(self, scenarios: pd.DataFrame) -> pd.DataFrame:
        """Make sure no scenario exceeds 3500 scooters in total.

        Args:
            scenarios: Output of a ProposalCleaner's create_scenarios.

        Returns:
            DataFrame with only those scenarios with an admissible
                number of scooters.
        """
        return (
            scenarios.groupby("scenario_id")
            .agg({"units": sum})
            .query("units <= 70")
            .drop("units", axis=1)
            .merge(scenarios, how="inner", on="scenario_id")
        )

    def optimize_consideration(
        self, admissible_scenarios: pd.DataFrame
    ) -> pd.DataFrame:
        """Keep only scenarios with the hightest total consideration.

        Args:
            admissible_scnearios: Output of filter_number_units

        Returns:
            Optimal scenarios w.r.t total consideration.
        """
        return (
            admissible_scenarios.assign(
                total_consideration=lambda df: df.units * df.consideration
            )
            .groupby("scenario_id")
            .agg({"total_consideration": sum})
            .assign(max_consideration=lambda df: df.total_consideration.max())
            .query("total_consideration==max_consideration")
            .drop(["max_consideration", "total_consideration"], axis=1)
            .merge(admissible_scenarios, how="inner", on="scenario_id")
        )

    def break_tie_priority(self, optimal_scenarios: pd.DataFrame) -> pd.DataFrame:
        """First tie breaker.

        Keeps the scenarios with the hightest number of
        proposals marked with number one priority.

        Args:
            optimal_scenarios: Output of optimize_consideration

        Returns:
            Those optimal scenarios in which the most proposals are
                marked with number one priority.
        """
        if optimal_scenarios.scenario_id.nunique() == 1:
            return optimal_scenarios
        return (
            optimal_scenarios.groupby("scenario_id")
            .agg({"priority": lambda x: x[x == 1].sum()})
            .assign(mas_unos=lambda df: df.priority.max())
            .query("priority==mas_unos")
            .drop(["mas_unos", "priority"], axis=1)
            .merge(optimal_scenarios, how="inner", on="scenario_id")
        )

    def break_tie_entropy(self, optimal_scenarios: pd.DataFrame) -> pd.DataFrame:
        """Second tie breaker.

        Keeps the scenario closest to the uniform
        distribution, in the sense of KL divergence.
        That is, the most entropic distirbution.

        Args:
            optimal_scenarios: Output of break_tie_priority

        Returns:
            The single optimal scenario w.r.t maximum entropy.
        """
        if optimal_scenarios.scenario_id.nunique() == 1:
            return optimal_scenarios
        return (
            optimal_scenarios.groupby("scenario_id")
            .agg({"units": wasserstein_to_uniform})
            .assign(min_wd=lambda df: df.units.min())
            .query("units==min_wd")
            .drop(["units", "min_wd"], axis=1)
            .merge(optimal_scenarios, how="inner", on="scenario_id")
        )

    def optimize(self, S: List[pd.DataFrame]) -> pd.DataFrame:
        """Get contest winners.

        Runs the total consideration optimizer and
        both tie breakers in the necessary case. See their
        respective documentation for details.


        Args:
            S: list of all possible scenarios.

        Returns:
            The winning strategy.
        """
        S = self.filter_number_units(S)
        S = self.optimize_consideration(S)
        S = self.break_tie_priority(S)
        S = self.break_tie_entropy(S)
        return S


class Contest:
    """A contest for determining which company will get the contracts.

    Attributes:
        cleaner: ProposalCleaner
        optimizer: ContestOptimizer
        optimal: winning strategy
    """

    def __init__(
        self, nagg: int = 1, nneu: int = 2, custom: ScooterCompany = None
    ) -> None:
        """Initialices participants for the contest.

        Args:
            nagg: Number of aggressive companies.
            nneu: Number of neutral companies.
            custom: A ScooterCompany object.
        """
        self.participants = self.create_participants(nagg, nneu, custom)
        self.cleaner = ProposalCleaner()
        self.proposals = None
        self.recieve_proposals()
        self.joint_proposals = pd.concat(self.proposals)
        self.optimizer = ContestOptimizer()
        self.optimal = None

    def create_participants(
        self, nagg: int, nneu: int, custom: ScooterCompany
    ) -> List[ScooterCompany]:
        """Create rival companies according to the given numbers.

        Args:
            nagg: Number of aggressive companies.
            nneu: Number of neutral companies.
            custom: A ScooterCompany object.
        Returns:
            List of participants with the given profiles.
        """
        neu = [NeutralScooterCompany() for _ in range(nneu)]
        agg = [AggressiveScooterCompany() for _ in range(nagg)]
        parts = neu + agg
        if custom is not None:
            parts += [custom]
        return parts

    def recieve_proposals(self) -> None:
        """Get list of DataFrames with proposals from each partipant.

        Returns:
            List of proposals to be checked by the optimizer.
        """
        props = [p.bid() for p in self.participants]
        self.proposals = props

    def get_winners(self) -> pd.DataFrame:
        """Run the optimizer.

        Returns:
            The contest's winning strategy.
        """
        if self.optimal is not None:
            return self.optimal
        clean_proposals = self.cleaner.create_scenarios(self.proposals)
        self.optimal = self.optimizer.optimize(clean_proposals)
        return self.optimal
