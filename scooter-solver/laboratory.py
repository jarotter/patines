from ax import RangeParameter, ParameterType, ParameterConstraint, SearchSpace
import numpy as np
from contest.utility_functions import IndependentUtilityFunction

class SearchSpaceFactory():
    def __init__(self, N):
        self.N = N

    def u(self, i):
        return RangeParameter(
            name = f"u{i}",
            parameter_type = ParameterType.INT,
            lower = 1, 
            upper = 35
        )
    
    def c(self, i):
        return RangeParameter(
                name = f"c{i}",
                parameter_type = ParameterType.FLOAT,
                lower = 1+1e-6, 
                upper = np.finfo(float).max
        )
    
    def D(self, i, j):
        return RangeParameter(
            name = f"D({i},{j})",
            parameter_type = ParameterType.INT,
            lower = 0, 
            upper = 1
        )
    
    def build_parameters(self):
        u = [self.u(i) for i in range(self.N)]
        c = [self.c(i) for i in range(self.N)]
        D = []
        for j in range(self.N):
            for i in range(j):
                D.append(self.D(i, j))
                
        return u + c + D
    
    def neq(self, i, j):
        M = self.N + 1
        c1 = ParameterConstraint(
            constraint_dict = {
                f"u{i}" : 1,
                f"u{j}" : -1,
                f"D({i},{j})" : -M
            },
            bound = -1
        )
        c2 = ParameterConstraint(
            constraint_dict = {
                f"u{i}" : -1,
                f"u{j}" : 1,
                f"D({i},{j})" : M
            },
            bound = M-1
        )
        return [c1, c2]

    def build_constraints(self):
        cons = []
        for j in range(self.N):
            for i in range(j):
                cons += self.neq(i, j)
        return cons

    def build(self):
        return SearchSpace(
            parameters = self.build_parameters(),
            parameter_constraints = self.build_constraints()
        )
