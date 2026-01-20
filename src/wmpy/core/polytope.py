from typing import Collection, Optional
import numpy as np
from pysmt.environment import Environment
from pysmt.fnode import FNode

from wmpy.core import Polynomial
from wmpy.core.inequality import Inequality

# from wmpy.optimization import ScipyOptimizer
import wmpy.optimization as opt
from wmpy.core.polynomial import PolynomialParser


class Polytope:
    """Internal class for convex H-polytopes.

    Attributes:
        inequalities: list of wmpy.core.Inequality
        variables: list of pysmt real variables
        env: the pysmt environment
        outer_box: the axis-aligned box (optional)
    """

    def __init__(
        self,
        expressions: Collection[FNode],
        variables: Collection[FNode],
        polynomials: PolynomialParser,
        env: Environment,
    ):
        """Default constructor for a H-polytope defined on an ordered list of variables (the continuous integration domain).

        Args:
           expressions: list of linear inequalities in pysmt format
           variables: list of pysmt real variables
           polynomials: common polynomial parser
           env: the pysmt environment
        """

        self.inequalities: list[Inequality] = []
        for expr in expressions:
            if expr.is_le() or expr.is_lt():
                self.inequalities.append(Inequality(expr, variables, polynomials, env))
            else:
                raise ValueError(f"Can't parse {expr}, not an (in)equality.")

        self.variables = variables
        self.polynomials = polynomials
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

        lowerl, upperl = [], []
        optimizer = opt.ScipyOptimizer()
        for i, var in enumerate(self.variables):
            cost = Polynomial(var, self.variables, self.polynomials, self.env)
            min_var = optimizer.optimize(self, cost, maximize=False)[i]
            max_var = optimizer.optimize(self, cost, maximize=True)[i]
            lowerl.append(min_var)
            upperl.append(max_var)

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
