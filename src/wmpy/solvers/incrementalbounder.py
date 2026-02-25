from typing import Any, Collection, Optional

import heapq
import numpy as np
from pysmt.fnode import FNode
import pysmt.shortcuts as smt

from wmpy.core import AssignmentConverter, Polytope, Polynomial
from wmpy.enumeration import Enumerator
from wmpy.integration import AxisAlignedWrapper, Integrator
from wmpy.optimization import CvxpyOptimizer

Box = tuple[np.ndarray, np.ndarray]

HeapEntry = tuple[
    float, int, tuple[Polytope, Polynomial, float, float, float, Box, Box]
]
Entry = tuple[float, Polytope, Polynomial, float, float, float, Box, Box]


class IncrementalBounder:
    """The class implements an incremental WMI bounding algorithm,
    which can be used to compute increasingly tight lower/upper bounds:

        Li <= Li+1 <= ... <= WMI(Delta, w) <= ... <= Ui+1 <= Ui

    The enumeration step is performed exhaustively once at
    instantiation.  The information on the convex fragments of the SMT
    theories are stored in a max-heap.
    """

    cp_epsilon = 1e-10
    cp_max_iter = 200
    max_tries = 5

    class DummyIntegrator(Integrator):
        """Used internally as a base integrator for the AxisAlignedWrapper
        (all integrals shoud be axis-aligned).
        """

        def integrate(self, polytope: Polytope, polynomial: Polynomial) -> float:
            raise RuntimeError("AxisAlignedWrapper didn't catch this.")

        def integrate_batch(
            self, convex_problems: Collection[tuple[Polytope, Polynomial]]
        ) -> np.ndarray:
            raise RuntimeError("AxisAlignedWrapper didn't catch this.")

    def __init__(self, enumerator: Enumerator, domain: Collection[FNode]):
        """Default constructor.

        Args:
            enumerator: the enumerator to use
            domain: the continuous integration domain (a list of pysmt real variables)
        """
        self.domain = domain
        self.enumerator = enumerator
        converter = AssignmentConverter(self.enumerator, self.domain)
        self.poly_parser = converter.poly_parser
        self.integrator = AxisAlignedWrapper(IncrementalBounder.DummyIntegrator())

        self.polys: list[HeapEntry] = []
        heapq.heapify(self.polys)

        self.lower_bound: float = 0.0
        self.upper_bound: float = 0.0
        for truth_assignment, nub in self.enumerator.enumerate(smt.Bool(True)):
            polytope, polynomial = converter.convert(truth_assignment)
            factor = 2**nub

            bounds = self._compute_local_bounds(polytope, polynomial)

            if bounds is None:
                raise RuntimeError("Could not compute initial bounds")

            ibox, obox, L, U = bounds
            diff_volume = (U - L) * factor
            self._insert_poly(
                diff_volume, polytope, polynomial, factor, L, U, ibox, obox
            )

            self.lower_bound += L * factor
            self.upper_bound += U * factor

    def refine(self) -> tuple[float, float]:
        """Tightens the global lower and upper bound by:

        - extracting from the heap the polytope with the largest difference between local lower/upper bounds
        - partitioning it into smaller polytopes using the previously computed inner box

        2N polytopes are added to the heap.

        Returns:
            A tuple containing the refined lower/upper bounds.
        """

        next_poly = self._extract_poly()
        # this could be empty, in that case it is not possible to further refine the bounds

        if next_poly is not None:
            _, polytope, polynomial, factor, L, U, ibox, obox = next_poly

            # refine the upper bound by first substituting it with the previous lower bound
            self.upper_bound += (L - U) * factor

            # break down the polytope into fragments
            for fragment in self._compute_fragments(polytope, ibox, obox):
                # print("frag:", fragment)

                bounds = self._compute_local_bounds(fragment, polynomial)

                if bounds is None:
                    # it was not possible to compute local bounds for the fragments
                    # it won't be considered again
                    continue

                f_ibox_poly, f_obox_poly, f_L, f_U = bounds
                f_diff_volume = (f_U - f_L) * factor
                self._insert_poly(
                    f_diff_volume,
                    fragment,
                    polynomial,
                    factor,
                    f_L,
                    f_U,
                    f_ibox_poly,
                    f_obox_poly,
                )

                self.lower_bound += f_L * factor
                self.upper_bound += f_U * factor

            # self.cp_epsilon /= 10
            # self.cp_max_iter *= 2

        return (self.lower_bound, self.upper_bound)

    def _insert_poly(
        self,
        diff: float,
        polytope: Polytope,
        polynomial: Polynomial,
        factor: float,
        L: float,
        U: float,
        ibox: Box,
        obox: Box,
    ) -> None:
        rest = polytope, polynomial, factor, L, U, ibox, obox
        entry = (-np.log(diff), id(rest), rest)
        heapq.heappush(self.polys, entry)

    def _extract_poly(self) -> Entry:
        max_entry = heapq.heappop(self.polys)
        return (max_entry[0], *max_entry[2])

    def _compute_local_bounds(
        self, polytope: Polytope, polynomial: Polynomial
    ) -> Optional[tuple[Box, Box, float, float]]:

        curr_epsilon = float(self.cp_epsilon)
        curr_max_iter = int(self.cp_max_iter)

        for i in range(self.max_tries):

            try:
                lower, upper, optimal = CvxpyOptimizer().compute_inner_box(
                    polytope, epsilon=curr_epsilon, max_iter=curr_max_iter
                )
            except RuntimeError:
                print("solver broke")
                raise RuntimeError

            tent_ibox = [lower, upper]
            obox = polytope.outer_box
            ibox = (
                np.maximum(obox[0], tent_ibox[0]),
                np.minimum(obox[1], tent_ibox[1]),
            )

            ibox_poly = self._box_to_polytope(ibox)
            obox_poly = self._box_to_polytope(obox)

            L = self.integrator.integrate(ibox_poly, polynomial)
            U = self.integrator.integrate(obox_poly, polynomial)

            if U - L > 0:
                return ibox, obox, L, U

            curr_epsilon /= 10
            curr_max_iter *= 2

            print(
                f"retrying {i} with epsilon: {self.cp_epsilon} and max_iter: {self.cp_max_iter}"
            )

        return None

    def _box_to_polytope(self, box: Box) -> Polytope:
        inequalities = []
        for i, var in enumerate(self.domain):
            inequalities.extend(
                [
                    smt.LE(smt.Real(float(box[0][i])), var),
                    smt.LE(var, smt.Real(float(box[1][i]))),
                ]
            )

        return Polytope(inequalities, self.poly_parser)

    def _compute_fragments(
        self, polytope: Polytope, ibox: Box, obox: Box
    ) -> list[Polytope]:

        obox_smt = self._box_to_polytope(obox).to_pysmt()
        fragments: list[Polytope] = []
        face_inequalities: list[FNode] = []

        # print("fragments!")
        # print("ibox\n", ibox, "\n")
        # print("obox\n", obox, "\n")
        for i, var in enumerate(self.domain):

            for side in [0, 1]:

                if abs(obox[side][i] - ibox[side][i]) < self.cp_epsilon:
                    # print(f"too tight i: {i} side: {side}", ibox[side][i], "vs", obox[side][i])

                    # bounds too tight to recurse
                    continue

                # convert current (outwards) face
                if bool(side):
                    ibox_face = smt.LE(smt.Real(float(ibox[side][i])), var)
                else:
                    ibox_face = smt.LE(var, smt.Real(float(ibox[side][i])))

                # fragment is made of:
                # (1) the ibox face
                fragment_inequalities = [ibox_face]

                # (2) the negated previous faces
                fragment_inequalities.extend(
                    [smt.Not(prev) for prev in face_inequalities]
                )

                face_inequalities.append(ibox_face)

                # (3) any other polytope inequality that is not redundant wrt aforementioned ibox faces
                for ineq in polytope.inequalities:
                    smt_ineq = ineq.to_pysmt()

                    # TODO: add redundancy checks
                    """
                    # an inequality is added if it's either tight wrt the outer box or not redundant
                    tight = not smt.is_sat(smt.And(smt_obox, smt.Not(smt_ineq)))
                    needed = smt.is_sat(
                        smt.And(smt_obox, *fragment_inequalities, smt.Not(smt_ineq))
                    )

                    # if tight or needed:
                    #    fragment_inequalities.append(smt_ineq)
                    """
                    fragment_inequalities.append(smt_ineq)

                fragment = Polytope(fragment_inequalities, self.poly_parser)
                fragments.append(fragment)

        return fragments
