from typing import Collection, Optional

import numpy as np
import pysmt.shortcuts as smt

from wmpy.core import Polynomial, Polytope
from wmpy.sampling import RejectionSampler


class RejectionIntegrator:
    """This class implements an integrator based on rejection sampling.
    The integral is approximated with its Monte Carlo (MC) estimate.
    """

    DEF_N_SAMPLES = int(10e3)

    def __init__(self, n_samples: Optional[int] = None, seed: Optional[int] = None):
        """Default constructor.

        Args:
            n_samples: sample size of the MC estimate
            seed: the seed number (optional)
        """
        self.n_samples = (
            RejectionIntegrator.DEF_N_SAMPLES if n_samples is None else n_samples
        )
        if seed is not None:
            np.random.seed(seed)

    def integrate(
        self, polytope: Polytope, integrand: Polynomial, max_iterations: int = 1
    ) -> float:
        """Computes a convex integral.

        Args:
            polytope: convex integration bounds
            polynomial: the integrand
            max_iterations: maximum number of rejection sampling attempts (default: 1)

        Returns:
            The result of the integration as a non-negative scalar value.
        """

        if integrand.is_zero:
            return 0.0

        uniform_weight = Polynomial(smt.Real(1), integrand.variables, integrand.env)
        lower, upper = polytope.compute_outer_box()
        sampler = RejectionSampler(polytope, uniform_weight)
        valid_sample = sampler.sample(self.n_samples, max_iterations=max_iterations)

        if len(valid_sample) > 0:
            # return the Monte Carlo estimate of the integral
            volume = (len(valid_sample) / self.n_samples) * np.prod(upper - lower)
            result = float(np.mean(integrand.to_numpy()(valid_sample)) * volume)
        else:
            result = 0.0

        return result

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
        for polytope, integrand in convex_integrals:
            volumes.append(self.integrate(polytope, integrand))

        return np.array(volumes)
