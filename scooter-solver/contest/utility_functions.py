from contest.strategies import CustomScooterCompany
from contest.solver import Contest
import numpy as np 
import pandas as pd

class IndependentUtilityFunction:
    """ Additive utility function as discussed in the slides.
    """
    
    def __init__(self, params, additive=True, rho=1/5):
        self.rho = rho
        self.params = params
        self.eval_func = self.additive if additive else self.multiplicative

        
    def __sample(self):
        """ Evaluate the point at one possible scenario.
        """
        sc = CustomScooterCompany(self.params)
        win = Contest(nagg=1, nneu=4, custom=sc).get_winners()
        e = win[win.units>0].shape[0]
        us = win[win.company==sc.name]
        u = us.units.values[0]
        if u==0:
            return -15
        c = us.consideration.values[0]

        return self.eval_func(u, e, c)
        
    def sample(self, size=500, ax=True):
        """ Evauate the point at more than one scenario.

        Parameters
        ----------
        size: int
            Number of samples to take at each point
        ax: bool
            Wether to return the entire sample or the output
            format required by Ax.
        """

        y = [self.__sample() for _ in range(size)]
        if not ax:
            return y
        return {
            "expected_utility": (np.mean(y), np.std(y))
        }

    
    def partial_consideration(self, c):
        """ Utility from consideration.
        
        Logarithmic and linear, piecewise.
        Parametrized by the Rappi Coefficient $\rho$.
        
        Parameters
        -----------
        c: float
            Unit consideration.
        """
        c_intercept = (self.rho+1)/self.rho
        if c<=c_intercept:
            return -np.log(self.rho*(c-1))
        return c_intercept - c
    
    def partial_market_presence(self, e):
        """
        """
        
        pmp = {
            2:2.0,
            3:1.0,
            4:0.25,
            5:0,
            6:0
        }
        return pmp[e]
    
    def partial_units(self, u):
        """ Utility from number of units.
        
        Parameter
        ---------
        u: int in [0, 35]
            Number of units. 0
        """
        
        return 1/35*u
    
    def additive(self, u, e, c):
        """ Adds partial utilities.
        """
        
        return (
            self.partial_units(u) +
            self.partial_market_presence(e) +
            self.partial_consideration(c)
        )
    
    def multiplicative(self, u, e, c):
        """ Multiplies partial utilities.
        """
        
        return (
            self.partial_units(u) *
            self.partial_market_presence(e) *
            self.partial_consideration(c)
        )
    
    