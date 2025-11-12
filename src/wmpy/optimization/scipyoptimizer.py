import numpy as np
from scipy.optimize import linprog
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from wmpy.core import Polynomial, Polytope


class ScipyOptimizer:

    def optimize(
        self, polytope: "Polytope", polynomial: "Polynomial", maximize: bool = True
    ) -> np.ndarray:
        """Solves the constrained optimization problem where:

        - polynomial is the objective function
        - polytope is the convex domain

        Either polynomial is maximized (default) or minimized.

        Args:
            polytope: convex integration bounds
            polynomial: the objective function
            maximize: Boolean flag (optional, def: True)

        Returns:
            A numpy array corresponding to the arg{max/min}.
        """

        obj = polynomial.to_numpy()
        A, B, S = polytope.to_numpy()
        N = len(polytope.variables)
        if polynomial.degree <= 1:  # solve LP
            coefficients = np.zeros(N)

            for i in range(N):
                ki = tuple(1 if j == i else 0 for j in range(N))
                coefficients[i] = polynomial.monomials.get(ki, 0)

            cost = -coefficients if maximize else coefficients
            res = linprog(
                cost,
                A_ub=A,
                b_ub=B,
                method="highs-ds",
                bounds=(None, None),
            )
            assert res.x is not None
            return res.x

        else:
            raise NotImplementedError(
                "Non-linear optimization is currently not supported"
            )
