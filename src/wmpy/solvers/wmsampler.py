from typing import Collection, Optional

import numpy as np

from pysmt.shortcuts import Bool
from pysmt.fnode import FNode

from wmpy.core import AssignmentConverter
from wmpy.enumeration import SAEnumerator
from wmpy.integration import LattEIntegrator
from wmpy.sampling import RejectionSampler


class WMSampler:
    """The class implements a sampling algorithm for weighted SMT formulas.

    The algorithm is divided:
    1) an offline preprocessing phase, enumerating the convex regions and computing their relative mass via WMI
    2) an online sampling phase, where the information above determines the different subsamples' sizes
    """

    def __init__(
        self,
        support: FNode,
        weight: Optional[FNode],
        domain: Collection[FNode],
        seed: Optional[int] = None,
    ):
        """Default constructor. Solves the preprocessing step by solving WMI.
        Uses this information for constructing a weighted strata for subsequent sampling efforts.

        Args:
            support: the support of the weight function (a pysmt formula)
            weights: the weight function as a pysmt term
            domain: the continuous integration domain (a list of pysmt real variables)
            seed: the seed number (optional)
        """
        enumerator = SAEnumerator(support, weight)
        converter = AssignmentConverter(enumerator)
        integrator = LattEIntegrator()
        convex_integrals = []
        n_unassigned_bools = []
        for truth_assignment, nub in enumerator.enumerate(Bool(True)):
            convex_integrals.append(converter.convert(truth_assignment, domain))
            n_unassigned_bools.append(nub)

        factors = [2**nb for nb in n_unassigned_bools]
        unnormalized_masses = integrator.integrate_batch(convex_integrals) * factors
        wmi: float = np.sum(unnormalized_masses)

        self.N = len(domain)
        self.subsamplers = []
        self.ratios: list[np.ndarray] = []
        for i, ci in enumerate(convex_integrals):
            self.subsamplers.append(RejectionSampler(*ci, seed))
            self.ratios.append(unnormalized_masses[i] / wmi)

        self.ratios = np.array(self.ratios)
        self.mult = np.ones(len(self.ratios))

    def sample(self, n_samples: int, max_iterations: int = 1) -> np.ndarray:
        """Draws a sample from (support, weight) in two phases.

        The subsample ratios are determined based on the strata precomputed at construction.
        Convex subsampling is carried over using rejection sampling.

        The procedure tries to satisfy the subsample size requirement up to `max_iterations`, returning M <= `n_samples` points.

        Args:
            n_samples: desired sample size
            max_iterations: maximum number of attempts (default: 1)

        Returns:
            A numpy array with shape (M, N).
        """
        required_size = np.max(
            [np.ones(len(self.subsamplers)), n_samples * self.ratios], axis=0
        ).astype(int)
        subsamples = [np.array([]).reshape((-1, self.N)) for _ in self.subsamplers]
        done = np.zeros(len(self.subsamplers)).astype(bool)
        it = 0
        while it < max_iterations:
            it += 1
            for i in range(len(self.subsamplers)):
                if len(subsamples[i]) < required_size[i]:
                    new_subsample = self.subsamplers[i].sample(
                        int(required_size[i] * self.mult[i])
                    )
                    self.mult[i] = required_size[i] / np.max([1, len(new_subsample)])
                    subsamples[i] = np.concatenate(
                        (subsamples[i], new_subsample), axis=0
                    )[: required_size[i]]
                    done[i] = len(subsamples[i]) == required_size[i]

            if done.all():
                break

        return np.concatenate(subsamples, axis=0)
