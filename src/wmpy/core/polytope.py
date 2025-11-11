from typing import Collection, Optional

import numpy as np
from pysmt.environment import Environment
from pysmt.fnode import FNode
from scipy.optimize import linprog

from wmpy.core.inequality import Inequality


class Polytope:
    """Internal class for convex H-polytopes.

    Attributes:
        inequalities: list of wmpy.core.Inequality
        N: the number of variables
        env: the pysmt environment
        outer_box: the axis-aligned box (optional)
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
        self.env = env
        self.outer_box: Optional[tuple[np.ndarray, np.ndarray]] = None

    def to_pysmt(self) -> FNode:
        """Returns a pysmt formula (FNode) encoding the polytope."""
        clauses = [ineq.to_pysmt() for ineq in self.inequalities]
        return self.env.formula_manager.And(*clauses)

    def compute_outer_box(self) -> tuple[np.ndarray, np.ndarray]:
        """Returns the tightest axis-aligned hyperrectangle fully
        enclosing the polytope by making 2N calls to an LP solver.

        The result is stored for future uses.

        Returns:
            Two numpy arrays corresponding to the extremes of the box.
        """
        if self.outer_box is not None:
            return self.outer_box

        A, B, _ = self.to_numpy()
        lowerl, upperl = [], []
        for i in range(self.N):
            cost = np.array([1 if j == i else 0 for j in range(self.N)])
            res = linprog(
                cost,
                A_ub=A,
                b_ub=B,
                method="highs-ds",
                bounds=(None, None),
            )
            assert res.x is not None
            lowerl.append(res.x[i])
            res = linprog(-cost, A_ub=A, b_ub=B, method="highs-ds", bounds=(None, None))
            assert res.x is not None
            upperl.append(res.x[i])

        self.outer_box = (np.array(lowerl), np.array(upperl))
        return self.outer_box

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
