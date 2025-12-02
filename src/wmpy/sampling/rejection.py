from typing import Callable, Collection, Optional

import numpy as np
from scipy.optimize import linprog

from wmpy.core import Polynomial, Polytope
from wmpy.optimization import ScipyOptimizer


class RejectionSampler:
    """This class implements rejection sampling from a polynomial distribution with convex support."""

    def __init__(
        self, polytope: Polytope, polynomial: Polynomial, seed: Optional[int] = None
    ) -> None:
        """Default constructor.

        Args:
            polytope: convex integration bounds
            polynomial: the target distribution
            seed: the seed number (optional)
        """

        if polynomial.is_zero:
            raise ValueError("Cannot sample from a null density")

        self.target = polynomial.to_numpy()
        w_argmax = ScipyOptimizer().optimize(polytope, polynomial)
        self.w_max = self.target(w_argmax.reshape((1, -1)))
        self.polytope = polytope

        if seed is not None:
            np.random.seed(seed)

    def sample(self, n_samples: int, max_iterations: int = 1) -> np.ndarray:
        """Draws a sample from a N-dimensional convex polytope using two-phases rejection.

        Initially, samples are uniformly sampled in the enclosing axis-aligned bounding box of the polytope.
        The first rejection phase discards points that are outside the polytope.
        Then, the second rejection phase ensures that samples are drawn from the target polynomial distribution.

        The procedure tries to sample `n_samples` points up to `max_iterations`, returning M <= `n_samples` points.

        Args:
            n_samples: desired sample size
            max_iterations: maximum number of attempts (default: 1)

        Returns:
            A numpy array with shape (M, N).
        """

        N = len(self.polytope.variables)
        result = np.array([]).reshape(-1, N)
        A, B, S = self.polytope.to_numpy()
        lower, upper = self.polytope.compute_outer_box()
        it = 0
        while it < max_iterations:
            it += 1
            uniform_sample = np.random.random((n_samples, N)) * (upper - lower) + lower
            valid_sample = uniform_sample[
                np.all(
                    (uniform_sample @ A[S].T < B[S])
                    & (uniform_sample @ A[~S].T <= B[~S]),
                    axis=1,
                )
            ]

            u = np.random.random(len(valid_sample)) * self.w_max
            valid_sample = valid_sample[u <= self.target(valid_sample)]
            result = np.concatenate((result, valid_sample), axis=0)

            if result.shape[0] >= n_samples:
                break

        return result[:n_samples, :]
