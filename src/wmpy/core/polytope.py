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
        remove_redundancies: bool=False,
    ):
        """Default constructor for a H-polytope defined on an ordered list of variables (the continuous integration domain).

        Args:
           expressions: list of linear inequalities in pysmt format
           variables: the continuous integration domain
           env: the pysmt environment
           remove_redundancies: use LP to remove redundant inequalities
        """
        self.inequalities:list[Inequality] = []
        for e in expressions:
            ineq = Inequality(e, variables, env)
            if not remove_redundancies or len(self.inequalities) == 0:
                self.inequalities.append(ineq)
            else:
                An, As, bn, bs = self.to_numpy()
                S, t = ineq.to_numpy()

        self.N = len(variables)
        self.mgr = env.formula_manager

    def to_pysmt(self) -> FNode:
        """Returns a pysmt formula (FNode) encoding the polytope."""
        if not self.inequalities:
            return self.mgr.Bool(True)
        return self.mgr.And(*map(lambda x: x.to_pysmt(), self.inequalities))

    def to_numpy(self, ignore_strictness:bool=False) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Converts the polytope to a tuple of numpy arrays.

        Args:
            ignore_strictness: if True discards information on strictness

        Returns:
            Four numpy arrays An, bn, As, bs encoding the polytope

              An x <= bn
              As x < bs

            If ignore_strictness is True As and bs are empty and

              An x {<=,<} bn
        """
        An, bn, As, bs = [], [], [], []
        for ineq in self.inequalities:
            A, b = ineq.to_numpy()
            if ignore_strictness or not ineq.strict:
                An.append(A)
                bn.append(b)
            else:
                As.append(A)
                bs.append(b)

        return np.array(An), np.array(bn), np.array(As), np.array(bs)

    def __str__(self) -> str:
        return "\n".join(["[" + str(b) + "]" for b in self.inequalities])
