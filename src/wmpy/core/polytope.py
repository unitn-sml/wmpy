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
           remove_redundancies: use LP to remove redundant inequalities
        """
        self.inequalities = [Inequality(e, variables, env) for e in expressions]
        self.N = len(variables)
        self.mgr = env.formula_manager

    def to_pysmt(self) -> FNode:
        """Returns a pysmt formula (FNode) encoding the polytope."""
        if not self.inequalities:
            return self.mgr.Bool(True)
        return self.mgr.And(*map(lambda x: x.to_pysmt(), self.inequalities))

    def to_numpy(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Converts the polytope to a tuple of numpy arrays.

        Returns:
            Three numpy arrays A, b, s:

              A x {<=,<} b

            s is a {0,1} array indicating which rows/entries in A, b correspond to strict inequalities.
        """
        A, b, s = [], [], []
        for ineq in self.inequalities:
            Ab = ineq.to_numpy()
            A.append(Ab[0])
            b.append(Ab[1])
            s.append(1 if ineq.strict else 0)

        return np.array(A), np.array(b), np.array(s)

    def __str__(self) -> str:
        return "\n".join(["[" + str(b) + "]" for b in self.inequalities])
