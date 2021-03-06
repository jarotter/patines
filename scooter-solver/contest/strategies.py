import numpy as np
import pandas as pd
from scipy.stats import gamma

word_file = "/usr/share/dict/words"
with open(word_file, 'r') as f:
    WORDS = pd.Series(f.read().splitlines(), dtype=str)
WORDS = WORDS[WORDS.str.len() == 4].reset_index(drop=True)
WORDS = WORDS[WORDS.str.lower() == WORDS].to_list()

class ScooterCompany:
    def __init__(self, name=None):
        self.name = self.get_random_name() if name is None else name
        
    
    def get_random_name(self):
        """ Get four letter word.
        """
        w = np.random.randint(0, len(WORDS))
        return WORDS[w]

class NeutralScooterCompany(ScooterCompany):
    def __init__(self, name=None):
        super().__init__(name)
        self.strategy = self.bid()

    def bid(self):
        """
        """
        N = np.random.randint(low=1, high=21)  
        unit_opts = np.arange(1, 35)
        unit_probs = 21 - unit_opts
        unit_probs = unit_probs/unit_probs.sum()
        units = np.random.choice(unit_opts, size=N, replace=False)
        consideration = 1 + gamma.rvs(a=6, scale=3/4, size=N)
        return pd.DataFrame({
            'company':self.name,
            'units': units,
            'consideration': consideration,
            'priority': np.arange(1, N+1)
        })

class AggressiveScooterCompany(ScooterCompany):
    def __init__(self, name=None):
        super().__init__(name)
        self.strategy = self.bid()

    def bid(self):
        """
        """
        N_opts = np.arange(1, 21)
        N_probs = np.power(21 - N_opts, 2)
        N_probs = N_probs/N_probs.sum()
        N = np.random.choice(N_opts, p=N_probs)
        units = [35-i for i in range(N)]
        consideration = 1 + gamma.rvs(a=12, scale=3/4, size=N)
        return pd.DataFrame({
            'company':self.name,
            'units': units,
            'consideration': consideration,
            'priority': np.arange(1, N+1)
        })

class CustomScooterCompany(ScooterCompany):
    def __init__(self, parametrization, name=None):
        super().__init__(name)
        self.strategy = self.build_strategy(parametrization)
        
    def build_strategy(self, params):
        units = []
        consideration = []
        for k, v in params.items():
            if 'u' in k:
                units.append(v)
            elif 'c' in k:
                consideration.append(v)
        assert len(units) == len(consideration)
        N = len(units)
        return pd.DataFrame({
            'company':self.name,
            'units': units,
            'consideration': consideration,
            'priority': np.arange(1, N+1)
        })

