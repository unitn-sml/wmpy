from typing import Collection, Optional, TYPE_CHECKING

import numpy as np

from wmpy.core import Polynomial, Polytope
from wmpy.core.inequality import Inequality

if TYPE_CHECKING:
    from wmpy.integration import Integrator


class AxisAlignedWrapper:
    """This class implements an integration wrapper for efficiently handling axis-aligned integration bounds.

    The enclosed integrator is called whenever the problem doesn't fall into this subcase.
    """

    def __init__(self, integrator: "Integrator"):
        """Default constructor.

        Args:
            integrator: the enclosed integrator instance
        """
        self.integrator = integrator

    def integrate(self, polytope: Polytope, polynomial: Polynomial) -> float:
        """Computes a convex integral.

        If the integration bounds are axis-aligned, the integral is computed in closed form.

        Args:
            polytope: convex integration bounds (a Polytope)
            polynomial: integrand (a Polynomial)

        Returns:
            The result of the integration as a non-negative scalar value.
        """
        intervals = AxisAlignedWrapper._axis_aligned_bounds(polytope)
        if intervals is not None:
            cumulative_integral = 0.0
            for exponents, coefficient in polynomial.monomials.items():
                monomial_integral = coefficient
                for i in range(polytope.N):
                    li, ui = intervals[i]
                    expi = exponents[i] + 1
                    if expi != 0:
                        monomial_integral *= (ui**expi - li**expi) / expi

                cumulative_integral += monomial_integral

            return cumulative_integral

        return self.integrator.integrate(polytope, polynomial)

    def integrate_batch(
        self, convex_integrals: Collection[tuple[Polytope, Polynomial]]
    ) -> np.ndarray:
        """Computes a batch of integrals.

        Args:
            convex_integrals: a collection of bounds/integrand pairs

        Returns:
            The result of the batch of integrations as a numpy array.
        """
        volumes = []
        for polytope, polynomial in convex_integrals:
            volumes.append(self.integrate(polytope, polynomial))

        return np.array(volumes)

    @staticmethod
    def _axis_aligned_bounds(polytope: Polytope) -> Optional[np.ndarray]:

        def parse_bound(inequality: Inequality) -> Optional[tuple[int, list[float]]]:
            monos = inequality.polynomial.monomials
            if len(monos) == 1:
                exp, coeff = next(iter(monos.items()))
                bound = [-np.inf, 0] if coeff > 0 else [0, np.inf]
                return exp.index(1), bound
            elif len(monos) == 2:
                ckey, exp = inequality.polynomial.ordered_keys
                const = monos[ckey]
                coeff = monos[exp]
                bound = (
                    [-np.inf, -const / coeff] if coeff > 0 else [-const / coeff, np.inf]
                )
                return exp.index(1), bound
            else:
                return None

        bounds = [[-np.inf, np.inf] for _ in range(polytope.N)]

        for ineq in polytope.inequalities:
            nvb = parse_bound(ineq)
            if nvb is None:
                return None
            nvar, new_bound = nvb
            old_bound = bounds[nvar]
            bounds[nvar] = [
                max(old_bound[0], new_bound[0]),
                min(old_bound[1], new_bound[1]),
            ]

        barray = np.array(bounds)
        if (barray[:, 0] > -np.inf).all() and (barray[:, 1] < np.inf).all():
            return barray
        else:
            return None
