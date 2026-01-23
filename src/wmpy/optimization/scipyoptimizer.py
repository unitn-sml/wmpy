import numpy as np
from scipy.linalg import block_diag
from scipy.optimize import linprog, minimize, LinearConstraint
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from wmpy.core import Polynomial, Polytope


class ScipyOptimizer:

    def __init__(self, epsilon: float = 1e-2) -> None:
        """Default constructor.

        Args:
            epsilon: small float constant used to enforce strict inequality constraints
        """
        self.epsilon = epsilon

    def compute_inner_box(self, polytope: "Polytope") -> tuple[np.ndarray, np.ndarray]:
        """Returns the largest axis-aligned hyperrectangle fully
        enclosed in the polytope by solving the convex optimization
        problem on 2N variables described here:

            https://scicomp.stackexchange.com/a/26465

        The result is stored for future uses.

        Args:
            polytope: the Polytope instance

        Returns:
            Two numpy arrays corresponding to the extremes of the box.

        """

        A, B, S = polytope.to_numpy()
        N = len(polytope.variables)

        print("A:")
        print(A.astype(float))
        print("b:")
        print(B.astype(float))

        l0 = linprog(np.ones(N), A_ub=A, b_ub=B, method="highs-ds").x
        Anotl0 = np.concatenate((A, -np.identity(N)))
        Bnotl0 = np.concatenate((B, l0 - np.ones(N) * self.epsilon))
        u0 = linprog(-np.ones(N), A_ub=Anotl0, b_ub=Bnotl0, method="highs-ds").x
        lu0 = np.concatenate((l0, u0))

        Al = -np.maximum(-A, np.zeros(A.shape))
        Au = np.maximum(A, np.zeros(A.shape))

        # U - L >= self.epsilon
        Aextra = np.concatenate((np.identity(N), -np.identity(N)), axis=1)
        Bextra = np.ones(N) * (-self.epsilon)

        # L <= U
        # Aextra = np.concatenate((Aextra, Aextra))
        # Bextra = np.concatenate((Bextra, np.zeros(N)))

        Atot = np.concatenate((block_diag(Al, Au), Aextra))
        Btot = np.concatenate((B, B, Bextra))

        constr = LinearConstraint(Atot, ub=Btot, keep_feasible=True)
        obj = lambda lu: -np.pow(np.prod(np.abs(lu[N:] - lu[:N])), 1 / N)
        # obj = lambda lu : -np.sum(np.abs(lu[N:] - lu[:N]))
        obj = lambda lu: -np.prod(lu[N:] - lu[:N])
        obj = lambda lu: -np.sum(lu[N:])

        print("Atot:")
        print(Atot.astype(float))
        print("btot:")
        print(Btot.astype(float))

        print("\n2LP x FINDING INITIAL (l,u)")
        print("l0:", lu0[:N], "u0:", lu0[N:])

        res = minimize(obj, lu0, method="trust-constr", constraints=constr)

        assert res.x is not None

        print("\nCONVEX OPTIMIZATION")
        print("message:", res.message)
        print("n.iters:", res.nit)
        print("l*:", res.x[:N], "u*:", res.x[N:])

        raise NotImplementedError("TOFIX")
        # return (res.x[:N], res.x[N:])

    def compute_outer_box(self, polytope: "Polytope") -> tuple[np.ndarray, np.ndarray]:
        """Returns the smallest axis-aligned hyperrectangle fully
        enclosing the polytope by making 2N calls to an LP solver.

        The result is stored for future uses.

        Args:
            polytope: the Polytope instance

        Returns:
            Two numpy arrays corresponding to the extremes of the box.
        """

        lowerl: list[float] = []
        upperl: list[float] = []
        A, B, S = polytope.to_numpy()
        N = len(polytope.variables)
        for i, var in enumerate(polytope.variables):
            cost = np.zeros(N)
            cost[i] = 1
            for sign, result_array in [(+1, lowerl), (-1, upperl)]:
                res = linprog(
                    sign * cost,
                    A_ub=A,
                    b_ub=B,
                    method="highs-ds",
                    bounds=(None, None),
                )
                assert res.x is not None
                result_array.append(res.x[i])

        return (np.array(lowerl), np.array(upperl))

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
            # TODO
            raise NotImplementedError(
                "Non-linear optimization is currently not supported"
            )
