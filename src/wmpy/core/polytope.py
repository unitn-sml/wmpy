from typing import Collection, Optional
import numpy as np
from pysmt.environment import Environment
from pysmt.fnode import FNode

from wmpy.core.inequality import Inequality
from wmpy.core.polynomial import PolynomialParser
import wmpy.optimization as opt


class Polytope:
    """Internal class for convex H-polytopes.

    Attributes:
        inequalities: list of wmpy.core.Inequality
        variables: list of pysmt real variables
        env: the pysmt environment
        inner_box: the largest enclosed axis-aligned box (optional)
        outer_box: the smallest enclosing axis-aligned box (optional)
    """

    def __init__(self, expressions: Collection[FNode], parser: PolynomialParser):
        """Default constructor for a H-polytope defined on an ordered list of variables (the continuous integration domain).

        Args:
            expressions: list of linear inequalities in pysmt format
            parser: a polynomial parser instance
        """

        self.inequalities: list[Inequality] = []
        for expr in expressions:
            self.inequalities.append(Inequality(expr, parser))

        self.variables = parser.variables
        self.env = parser.env
        self._inner_box: Optional[tuple[np.ndarray, np.ndarray]] = None
        self._outer_box: Optional[tuple[np.ndarray, np.ndarray]] = None

    @property
    def inner_box(self) -> tuple[np.ndarray, np.ndarray]:
        if self._inner_box is None:
            lower, upper, _ = opt.CvxpyOptimizer().compute_inner_box(self)
            self._inner_box = lower, upper

        return self._inner_box

    @property
    def outer_box(self) -> tuple[np.ndarray, np.ndarray]:
        if self._outer_box is None:
            self._outer_box = opt.ScipyOptimizer().compute_outer_box(self)

        return self._outer_box

    def to_pysmt(self) -> FNode:
        """Returns a pysmt formula (FNode) encoding the polytope."""
        clauses = [ineq.to_pysmt() for ineq in self.inequalities]
        return self.env.formula_manager.And(*clauses)

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
