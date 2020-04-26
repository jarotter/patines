"""Bidding strategies for scooter companies."""
from abc import abstractmethod
from platform import system
from typing import Dict, List

import numpy as np
import pandas as pd
from scipy.stats import gamma


class Namer:
    """Random name generator.

    Attributes:
        os: The running operating system
        words: A list of words
    """

    def __init__(self) -> None:
        self.os = system()
        self.words = self.get_word_list()

    def get_word_list(self) -> List[str]:
        """Get random word list.

        If running an unix-like system, it will choose all of the
        four letter words in `/usr/share/dict/words`. Otherwise it
        will just use a combination of the string 'cp' and two random digits.
        """
        if self.os == "Windows":
            return ["cp" + str(i).zfill(2) for i in range(100)]
        else:
            WORD_FILE = "/usr/share/dict/words"
            with open(WORD_FILE, "r") as f:
                WORDS = pd.Series(f.read().splitlines(), dtype=str)
            WORDS = WORDS[WORDS.str.len() == 4].reset_index(drop=True)
            WORDS = WORDS[WORDS.str.lower() == WORDS].to_list()
            return WORDS

    def get_random_name(self) -> str:
        """Get four letter word from the list of words."""
        w = np.random.randint(0, len(self.words))
        return self.words[w]


class ScooterCompany:
    """A scooter company."""

    def __init__(self, name: str = None) -> None:
        self.name = Namer().get_random_name() if name is None else name

    @abstractmethod
    def bid(self, N: int) -> pd.DataFrame:
        """Generate proposal."""
        pass


class UniformScooterCompany(ScooterCompany):
    r"""A scooter company that bids uniformly at random.

    More formally, the generating process for this company is

        N ~ Uniform{1, ..., 20}
        c_i ~ 1 + Gamma(6, 3/4) for i = 1, ..., N
        n_1 ~ Uniform{1, ..., 35}
        n_i ~ Uniform{1, ..., 35}\{n_j: j < 1} for i = 2, ..., N
    """

    def __init__(self, name: str = None) -> None:
        super().__init__(name)

    def bid(self, N: int = None) -> pd.DataFrame:
        """Construct proposal."""
        if N is None:
            N = np.random.randint(low=1, high=21)
        unit_opts = np.arange(1, 35)
        units = np.random.choice(unit_opts, size=N, replace=False)
        consideration = 1 + gamma.rvs(a=6, scale=3 / 4, size=N)
        return pd.DataFrame(
            {
                "company": self.name,
                "units": units,
                "consideration": consideration,
                "priority": np.arange(1, N + 1),
            }
        )


class NeutralScooterCompany(ScooterCompany):
    """A scooter company with neutral strategy.

    This companies slightly prefer options with more units,
    but still are willing to bid a large range of considerations.

    Formally, the generating process they follow
        N ~ Uniform{1, ..., 20}
        p_i proportional to (35-i) for i = 1, ..., 35
        {u_i : i = 1, ..., N} is a sample taken without replacement from {1, ..., N}
            with probabilities (p_1, ..., p_N)
        c_i ~ 1 + Gamma(6, 3/4) for i = 1, ..., N
    """

    def __init__(self, name: str = None) -> None:
        super().__init__(name)

    def bid(self, N: int = None) -> pd.DataFrame:
        """Construct proposal."""
        if N is None:
            N = np.random.randint(low=1, high=21)
        unit_opts = np.arange(1, 35)
        unit_probs = 35 - unit_opts
        unit_probs = unit_probs / unit_probs.sum()
        units = np.random.choice(unit_opts, size=N, replace=False, p=unit_probs)
        consideration = 1 + gamma.rvs(a=6, scale=3 / 4, size=N)
        return pd.DataFrame(
            {
                "company": self.name,
                "units": units,
                "consideration": consideration,
                "priority": np.arange(1, N + 1),
            }
        )


class AggressiveScooterCompany(ScooterCompany):
    """A scooter company with aggressive strategy.

    This companies strongly prefer options with more units
    and will bid high considerations.

    Formally, the generating process they follow is
        q_i proportional to (21-i)^2 for i = 1, ..., 20
        N ~ Categorical_{20}(q_1, ..., q_20)
        p_i proportional to (35-i) for i = 1, ..., 35
        u_i = 36 - i for i = 1, ..., N
        c_i ~ 1 + Gamma(12, 3/4) for i = 1, ..., N
    """

    def __init__(self, name: str = None) -> None:
        super().__init__(name)

    def bid(self, N: int = None) -> pd.DataFrame:
        """Generate proposal."""
        if N is None:
            N_opts = np.arange(1, 21)
            N_probs = np.power(21 - N_opts, 2)
            N_probs = N_probs / N_probs.sum()
            N = np.random.choice(N_opts, p=N_probs)
        units = [35 - i for i in range(N)]
        consideration = 1 + gamma.rvs(a=12, scale=3 / 4, size=N)
        return pd.DataFrame(
            {
                "company": self.name,
                "units": units,
                "consideration": consideration,
                "priority": np.arange(1, N + 1),
            }
        )


class CustomScooterCompany(ScooterCompany):
    """Scooter company that bids deterministically."""

    def __init__(self, params: Dict, name: str = None) -> None:
        super().__init__(name)
        self.params = params

    def bid(self) -> pd.DataFrame:
        """Generate dataframe with given bid."""
        N = len(self.params["units"])
        M = len(self.params["consideration"])
        assert N == M
        return pd.DataFrame(
            {
                "company": self.name,
                "units": self.params["units"],
                "consideration": self.params["consideration"],
                "priority": np.arange(1, N + 1),
            }
        )
