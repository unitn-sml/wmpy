from typing import Collection, Optional
import numpy as np
from pysmt.environment import Environment
from pysmt.fnode import FNode

from wmpy.core import Polynomial
from wmpy.core.inequality import Inequality
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

    def __init__(
        self,
        expressions: Collection[FNode],
        variables: Collection[FNode],
        env: Environment,
    ):
        """Default constructor for a H-polytope defined on an ordered list of variables (the continuous integration domain).

        Args:
           expressions: list of linear inequalities in pysmt format
           variables: list of pysmt real variables
           env: the pysmt environment
        """

        self.inequalities: list[Inequality] = []
        for expr in expressions:
            if expr.is_le() or expr.is_lt():
                self.inequalities.append(Inequality(expr, variables, env))
            else:
                raise ValueError(f"Can't parse {expr}, not an (in)equality.")

        self.variables = variables
        self.env = env
        self._inner_box: Optional[tuple[np.ndarray, np.ndarray]] = None
        self._outer_box: Optional[tuple[np.ndarray, np.ndarray]] = None

    @property
    def inner_box(self) -> tuple[np.ndarray, np.ndarray]:
        if self._inner_box is None:
            # self._inner_box = opt.ScipyOptimizer().compute_inner_box(self)
            self._inner_box = opt.CvxpyOptimizer().compute_inner_box(self)

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
