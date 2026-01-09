import cvxpy as cp
import numpy as np
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from wmpy.core import Polynomial, Polytope


class CvxpyOptimizer:

    DEF_SOLVER = "SCS"

    def __init__(self, epsilon=1e-2):
        """Default constructor.

        Args:
            epsilon: small float constant used to enforce strict inequality constraints
        """
        self.epsilon = epsilon


    def compute_inner_box(self, polytope) -> tuple[np.ndarray, np.ndarray]:
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
        m, n = A.shape

        Aplus = np.maximum(np.zeros(A.shape), A)
        Aminus = np.maximum(np.zeros(A.shape), -A)

        l = cp.Variable(n)
        u = cp.Variable(n)

        obj = -cp.geo_mean(u - l)
        minimizer = cp.Minimize(obj)
        constraints = [Aplus @ u - Aminus @ l <= B]
        prob = cp.Problem(minimizer, constraints)
        
        l1dist = -prob.solve(solver="SCS", eps=1e-8)

        if np.abs(l1dist) == np.inf:
            print("Unbounded problem")
        else:
            print(f"L1 dist: {l1dist}")
            print(f"l: {l.value}")
            print(f"u: {u.value}")
            vol = np.prod(u.value - l.value)
            print(f"volume: {vol}")

        return (l.value, u.value)


