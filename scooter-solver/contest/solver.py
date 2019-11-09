import numpy as np
import pandas as pd
from itertools import product
from scipy.stats import entropy
from contest.strategies import NeutralScooterCompany, AggressiveScooterCompany

class ProposalCleaner:
    def __init__(self):
        pass
    
    def add_null_proposals(self, proposals):
        """ Adds the null 0 units proposal to each participant.
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
    n = p.shape[0]
    p = p/p.sum()
    return np.log2(n) - entropy(p)

class ContestOptimizer:
    def __init__(self):
        pass
    
    def filter_number_units(self, scenarios):
        return (
            scenarios
            .groupby('scenario_id')
            .agg({'units':sum})
            .query('units <= 70')
            .drop('units', axis=1)
            .merge(scenarios, how='inner', on='scenario_id')
        )
    
    def optimize_number_units(self, admissible_scenarios):
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
        S = self.filter_number_units(S)
        S = self.optimize_number_units(S)
        S = self.break_tie_priority(S)
        S = self.break_tie_entropy(S)
        return S

class Contest:
    def __init__(self, nagg=1, nneu=2, custom=None):
        self.participants = self.create_participants(nagg, nneu, custom)
        self.cleaner = ProposalCleaner()
        self.proposals = self.recieve_proposals()
        self.optimizer = ContestOptimizer()
        self.optimal = None
        
    
    def create_participants(self, nagg, nneu, custom):
        """ According to the numbers given.
        """
        neu = [NeutralScooterCompany() for _ in range(nneu)]
        agg = [AggressiveScooterCompany() for _ in range(nagg)]
        parts = neu + agg
        if custom is not None:
            parts += [custom]
        return parts
    
    def recieve_proposals(self):
        """ Get list of DataFrames with the proposals.
        """
        return [p.strategy for p in self.participants]
    
    
    
    def get_winners(self):
        if self.optimal is not None:
            return self.optimal
        clean_proposals = self.cleaner.create_scenarios(self.proposals)
        self.optimal = self.optimizer.optimize(clean_proposals)
        return self.optimal

