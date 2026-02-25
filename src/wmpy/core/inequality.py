from typing import Any, Collection

import numpy as np
from pysmt.fnode import FNode
from pysmt.shortcuts import LE, LT

from wmpy.core.polynomial import PolynomialParser


class Inequality:
    """Internal class for inequalities in the canonical form:

        P {<,<=} 0

    where P is a degree 1 polynomial.

    Attributes:
        strict: boolean flag, true when the inequality is <
        mgr: pysmt formula manager
        polynomial: the Polynomial P
    """

    def __init__(self, expr: FNode, parser: PolynomialParser):
        """Default constructor.

        Args:
            expr: the inequality in pysmt format
            parser: a polynomial parser instance
        """
        if expr.is_le() or expr.is_lt():
            self.strict = expr.is_lt()
        elif expr.is_not() and (expr.args()[0].is_le() or expr.args()[0].is_lt()):
            neg_ineq = expr.args()[0]
            first, second = neg_ineq.args()
            self.strict = neg_ineq.is_le()
            if neg_ineq.is_le():
                expr = LT(second, first)
            else:
                expr = LE(second, first)
        else:
            raise ValueError("Not an inequality")

        self.mgr = parser.env.formula_manager

        p1, p2 = expr.args()
        # (p1 OP p2) => (p1 - p2 OP 0)
        poly_sub = self.mgr.Plus(p1, self.mgr.Times(self.mgr.Real(-1), p2))
        self.polynomial = parser.parse(poly_sub)
        if self.polynomial.degree != 1:
            raise ValueError("Non-linear polynomial in inequality")

    def to_numpy(self) -> tuple[np.ndarray, float]:
        """Converts the inequality in numpy format:

            A {<=,<} b

        Returns:
            A numpy array A, a scalar b
        """
        N = len(self.polynomial.variables)
        const_key = tuple(0 for _ in range(N))
        key = lambda i: tuple(1 if j == i else 0 for j in range(N))
        A = [self.polynomial.monomials.get(key(i), 0) for i in range(N)]
        b = -self.polynomial.monomials.get(const_key, 0)
        return np.array(A), b

    def to_pysmt(self) -> FNode:
        """Converts the inequality in pysmt format.

        Returns:
            Either a LT (less-than) or LE (less-or-equal-than) atom.
        """
        op = self.mgr.LT if self.strict else self.mgr.LE
        return op(self.polynomial.to_pysmt(), self.mgr.Real(0))

    def __str__(self) -> str:
        opstr = "<" if self.strict else "<="
        return f"({str(self.polynomial)}) {opstr} 0"

    def __eq__(self, other: Any) -> bool:
        # TODO: this shouldn't be needed
        raise NotImplementedError()

    def __hash__(self) -> int:
        # TODO: this should be needed
        # return hash(self.constant)
        raise NotImplementedError()
