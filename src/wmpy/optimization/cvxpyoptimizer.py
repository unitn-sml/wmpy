import cvxpy as cp
import numpy as np
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from wmpy.core import Polynomial, Polytope


class CvxpyOptimizer:

    def compute_inner_box(
        self, polytope: "Polytope", epsilon: float = 1e-15, max_iter: int = 200
    ) -> tuple[np.ndarray, np.ndarray, bool]:
        """Returns the largest axis-aligned hyperrectangle fully
        enclosed in the polytope by solving the convex optimization
        problem on 2N variables described here:

            https://scicomp.stackexchange.com/a/26465

        The result is stored for future uses.

        Args:
            polytope: the Polytope instance
            epsilon: tolerance parameter for convex optimization (def: 1e-15)
            max_iter: maximum number of iterations

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
        constraints = [A @ l <= B, A @ u <= B, Aplus @ u - Aminus @ l <= B]

        prob = cp.Problem(minimizer, constraints)

        try:
            prob.solve(
                solver="clarabel",
                max_iter=max_iter,
                tol_feas=epsilon,
                tol_gap_abs=epsilon,
                tol_gap_rel=epsilon,
                tol_infeas_abs=epsilon,
                tol_infeas_rel=epsilon,
                # tol_ktratio=epsilon,
                verbose=False,
            )
        except cp.error.SolverError:
            prob.solve(solver="scs")
            # raise RuntimeError("Cvxpy solver error")

        if np.abs(prob.value) == np.inf:
            raise ValueError("Unbounded problem")

        assert l.value is not None
        assert u.value is not None
        return l.value, u.value, prob.status == cp.OPTIMAL
