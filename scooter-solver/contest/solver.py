import numpy as np
import pandas as pd
from itertools import product
from scipy.stats import entropy
from contest.strategies import NeutralScooterCompany, AggressiveScooterCompany

class ProposalCleaner:
    """ Utilities for cleaning generated proposals.
    """

    def __init__(self):
        pass
    
    def add_null_proposals(self, proposals):
        """ Adds the null 0 units proposal to each company.

        Parameters
        -----------
        proposals: DataFrame
            With each company's proposal
        
        Returns
        -------
        list:
            Each company now has a proposal with 0 units as
            last priority.
        """

        new_proposals = []
            
        for p in proposals:
            N = p.shape[0]
            name = p.company.values[0]
            pp = p.append({
                'company': name,
                'units': 0,
                'consideration': 0,
                'priority': N+1
            }, ignore_index=True).reset_index(drop=True)
            new_proposals.append(pp)
        return new_proposals 
    
    def create_scenarios(self, proposals):
        """ Create combinations of companies.

        Cartesian product of every proposal, across
        all companies, creating a long dataframe in which 
        every combination is uniquely identified.

        Parameters
        -----------
        proposals: DataFrame
            List of proposals.

        Returns
        -------
        DataFrame
        """
    
        participants = [p.company.values[0] for p in proposals]
        priority_prod = (
            pd.DataFrame(
                product(*(p.priority for p in proposals)),
                columns=participants)
            .reset_index(drop=False)
            .rename(columns={'index':'scenario_id'})
            .melt(id_vars='scenario_id', var_name='company', value_name='priority')
        )
        
        props = pd.concat(self.add_null_proposals(proposals), sort=False)
        return priority_prod.merge(props, how='left', on=['company', 'priority'])

def kl_to_uniform(p):
    """ KL Divergence to the discrete uniform distribution.

    Parameters
    ----------
    p: ndarray
        Probability distribution. If not normalized, it will be.

    Returns
    -------
    float
    """
    n = p.shape[0]
    p = p/p.sum()
    return np.log2(n) - entropy(p)

class ContestOptimizer:
    def __init__(self):
        pass
    
    def filter_number_units(self, scenarios):
        """ Make sure no scenario exceeds 3500 scooters.

        Parameters
        ----------
        scenarios: DataFrame
            Output of a ProposalCleaner's create_scenarios.

        Returns
        -------
        DataFrame:
            With admissible scenarios only.
        """

        return (
            scenarios
            .groupby('scenario_id')
            .agg({'units':sum})
            .query('units <= 70')
            .drop('units', axis=1)
            .merge(scenarios, how='inner', on='scenario_id')
        )
    
    def optimize_number_units(self, admissible_scenarios):
        """ Keeps only scenarios with the hightest total consideration.

        Parameters
        ----------
        admissible_scenarios: DataFrame
            Output of filter_number_units

        Returns
        -------
        DataFrame:
            Optimal w.r.t total consideration.
        """

        return (
            admissible_scenarios
            .assign(total_consideration = lambda df: df.units*df.consideration)
            .groupby('scenario_id')
            .agg({'total_consideration':sum})
            .assign(max_consideration = lambda df: df.total_consideration.max())
            .query('total_consideration==max_consideration')
            .drop(['max_consideration', 'total_consideration'], axis=1)
            .merge(admissible_scenarios, how='inner', on='scenario_id')
        )
    
    def break_tie_priority(self, optimal_scenarios):
        """ First tie breaker.

        Keeps the scenarios with the hightest number of
        proposals marked with number one priority.

        Parameters
        ----------
        optimal_scenarios:
            Output of optimize_number_units
        
        Returns
        -------
        DataFrame
        """

        if optimal_scenarios.scenario_id.nunique() == 1:
            return optimal_scenarios
        return (
            optimal_scenarios
            .groupby('scenario_id')
            .agg({'priority': lambda x: x[x==1].sum()})
            .assign(mas_unos = lambda df: df.priority.max())
            .query('priority==mas_unos')
            .drop(['mas_unos', 'priority'], axis=1)
            .merge(optimal_scenarios, how='inner', on='scenario_id')
        )
    
    def break_tie_entropy(self, optimal_scenarios):
        """ Second tie breaker.

        Keeps the scenario closest to the uniform
        distribution, in the sense of KL divergence.
        That is, the most entropic distirbution.

        Parameters
        ----------
        optimal_scenarios:
            Output of break_tie_priority
        
        Returns
        -------
        DataFrame
        """
        
        if optimal_scenarios.scenario_id.nunique() == 1:
            return optimal_scenarios
        return (
            optimal_scenarios
            .groupby('scenario_id')
            .agg({'units': kl_to_uniform})
            .assign(min_kl = lambda df: df.units.min())
            .query('units==min_kl')
            .drop(['units', 'min_kl'], axis=1)
            .merge(optimal_scenarios, how='inner', on='scenario_id')
            
        )
    
    def optimize(self, S):
        """ Get contest winners.

        Runs the total consideration optimizer and
        both tie breakers in the necessary case. See their
        respective documentation for details.

        
        Parameters
        ----------
        S: list
            List of all possible scenarios.

        Returns
        -------
        DataFrame:
            With the winning strategies.
        """

        S = self.filter_number_units(S)
        S = self.optimize_number_units(S)
        S = self.break_tie_priority(S)
        S = self.break_tie_entropy(S)
        return S

class Contest:
    def __init__(self, nagg=1, nneu=2, custom=None):
        self.participants = None
        self.create_participants(nagg, nneu, custom)
        self.cleaner = ProposalCleaner()
        self.proposals = self.recieve_proposals()
        self.joint_proposals = self.recieve_proposals(joint=True)
        self.optimizer = ContestOptimizer()
        self.optimal = None

    def create_participants(self, nagg, nneu, custom):
        """ According to the numbers given, and sets 
        is as a property object.

        Parameters
        -----------
        nagg: nonnegative integer
            Number of aggressive companies.
        nneu: nonnegative integer
            Number of neutral companies.
        custom: CustomScooterCompany
            A ScooterCompany object to represent the optimizer's strategy.
        """
        neu = [NeutralScooterCompany() for _ in range(nneu)]
        agg = [AggressiveScooterCompany() for _ in range(nagg)]
        parts = neu + agg
        if custom is not None:
            parts += [custom]
        self.participants = parts
        return

    def recieve_proposals(self, joint=False):
        """ Get list of DataFrames with proposals from 
        each partipant.

        Parameters
        -----------
        joint: bool
            If True, returns a single DataFrame (the human readable
            version of the proposals)

        Returns
        --------
        List of DataFrames:
            To be used by the solver
        """

        props = [p.strategy for p in self.participants]
        if not joint:
            return props
        return pd.concat(props)
    
    
    
    def get_winners(self):
        """ Runs the optimizer.

        Returns
        --------
        DataFrame:
            With the contest's results.
        """

        if self.optimal is not None:
            return self.optimal
        clean_proposals = self.cleaner.create_scenarios(self.proposals)
        self.optimal = self.optimizer.optimize(clean_proposals)
        return self.optimal

