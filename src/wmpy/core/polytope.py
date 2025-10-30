from typing import Collection

import numpy as np
from pysmt.environment import Environment
from pysmt.fnode import FNode

from wmpy.core.inequality import Inequality


class Polytope:
    """Internal class for convex H-polytopes.

    Attributes:
        inequalities: list of wmpy.core.Inequality
        N: the number of variables
    """

    def __init__(
        self,
        expressions: Collection[FNode],
        variables: Collection[FNode],
        env: Environment,
    ):
        """Default constructor for a H-polytope defined on an ordered list of variables (the continuous integration domain).

        Args:
           expressions: list of linear inequalities in pysmt format
           variables: the continuous integration domain
           env: the pysmt environment
        """

        self.inequalities: list[Inequality] = []
        for expr in expressions:
            if expr.is_le() or expr.is_lt():
                self.inequalities.append(Inequality(expr, variables, env))
            else:
                raise ValueError(f"Can't parse {expr}, not an (in)equality.")

        self.N = len(variables)
        self.mgr = env.formula_manager

    def to_pysmt(self) -> FNode:
        """Returns a pysmt formula (FNode) encoding the polytope."""
        clauses = [ineq.to_pysmt() for ineq in self.inequalities]
        return self.mgr.And(*clauses)

    def to_numpy(
        self,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Converts the polytope to a tuple of numpy arrays.

        Returns:
            Three numpy arrays A, B, S encoding the polytope

              A x {<=/<} B

            S is a {0,1} array indicating which rows/entries in A, B correspond to strict inequalities.
        """
        A, B, S = [], [], []
        for ineq in self.inequalities:
            Ab = ineq.to_numpy()
            A.append(Ab[0])
            B.append(Ab[1])
            S.append(1 if ineq.strict else 0)

        return np.array(A), np.array(B), np.array(S)

    def __str__(self) -> str:
        return "\n".join(["[" + str(b) + "]" for b in self.inequalities])
