import numpy as np
import pysmt.shortcuts as smt
from pysmt.typing import REAL
from pysmt.walkers import IdentityDagWalker
import pytest

from wmpy.core import PolynomialParser

env = smt.get_env()
x = smt.Symbol("X", REAL)
y = smt.Symbol("Y", REAL)


class PowSimplifier(IdentityDagWalker):
    """This is needed here because wmpy follows python's convention
    on 0 ** 0 == 1. This is not the case for most SMT solvers.

    """

    def walk_pow(self, formula, args):
        b, e = args
        if e.is_real_constant(0.0):
            return smt.Real(1.0)
        else:
            return smt.Pow(b, e)

    def simplify(self, polynomial):
        return self.walk(polynomial)


powsimplifier = PowSimplifier()


def equivalent_expressions(original, derived):
    """Polynomial simplification might result in a pysmt expression
    with less variables. If that's not the case, use an SMT solver to
    check for equivalence.
    """
    return (
        original.get_free_variables() != derived.get_free_variables()
        or not smt.is_sat(smt.Not(smt.Equals(original, derived)))
    )


def same_numerical_constant(output_val, expected_val):
    """Numerical values after pysmt / Fraction conversions might not
    match floating point operations resulting in NaN or
    Inf. Otherwise, results are expected to be "close enough".

    """
    return (
        np.isnan(expected_val)
        or np.isinf(expected_val)
        or np.isclose(output_val, expected_val)
    )


def test_no_variables():
    with pytest.raises(ValueError):
        _ = PolynomialParser([], env)


def test_unknown_variables():
    with pytest.raises(ValueError):
        _ = PolynomialParser([x], env).parse(y)


def test_constant(f_const):
    expression = smt.Real(f_const)
    polynomial = PolynomialParser([x], env).parse(expression)
    assert equivalent_expressions(expression, polynomial.to_pysmt())
    coefficient = polynomial.monomials.get((0,), 0)
    assert same_numerical_constant(coefficient, f_const)


def test_constant_sum(f_vec2):
    try:
        c1, c2 = f_vec2
        expression = smt.Plus(*map(smt.Real, f_vec2))
        polynomial = PolynomialParser([x], env).parse(expression)
        assert equivalent_expressions(expression, polynomial.to_pysmt())
        coefficient = polynomial.monomials.get((0,), 0)
        assert same_numerical_constant(coefficient, c1 + c2)
    except OverflowError:
        pass


def test_constant_multiplication(f_vec2):
    try:
        c1, c2 = f_vec2
        expression = smt.Times(*map(smt.Real, f_vec2))
        polynomial = PolynomialParser([x], env).parse(expression)
        assert equivalent_expressions(expression, polynomial.to_pysmt())
        coefficient = polynomial.monomials.get((0,), 0)
        assert same_numerical_constant(coefficient, c1 * c2)
    except OverflowError:
        pass


def test_constant_power(f_const, exp_const):
    try:
        expression = smt.Pow(smt.Real(f_const), smt.Real(exp_const))
        polynomial = PolynomialParser([x], env).parse(expression)
        assert equivalent_expressions(expression, polynomial.to_pysmt())
        coefficient = polynomial.monomials.get((0,), 0)
        assert same_numerical_constant(coefficient, f_const**exp_const)
    except OverflowError:
        pass


def test_univariate_symbol():
    expression = x
    polynomial = PolynomialParser([x], env).parse(expression)
    assert equivalent_expressions(expression, polynomial.to_pysmt())
    coefficient = polynomial.monomials.get((1,), 0)
    assert coefficient == 1


def test_univariate_expression1(f_vec2, exp_vec2):
    c1, c2 = f_vec2
    exp1, exp2 = exp_vec2
    expression = smt.Plus(
        smt.Times(smt.Real(c1), smt.Pow(x, smt.Real(exp1))),
        smt.Times(smt.Real(c2), smt.Pow(x, smt.Real(exp2))),
    )
    polynomial = PolynomialParser([x], env).parse(expression)
    assert equivalent_expressions(
        powsimplifier.simplify(expression), polynomial.to_pysmt()
    )

    if exp1 == exp2 and (c1 + c2) == 0:
        assert polynomial.degree == 0
        assert polynomial.is_zero
    else:
        assert polynomial.degree == max(exp1 * int(c1 != 0), exp2 * int(c2 != 0))

    if exp1 == exp2:
        assert same_numerical_constant(polynomial.monomials.get((exp1,), 0), c1 + c2)

    else:
        if exp1 != 0:
            assert same_numerical_constant(polynomial.monomials.get((exp1,), 0), c1)

        if exp2 != 0:
            assert same_numerical_constant(polynomial.monomials.get((exp2,), 0), c2)

    assert same_numerical_constant(
        polynomial.monomials.get((0,), 0),
        int(exp1 == 0) * c1 + int(exp2 == 0) * c2,
    )


def test_univariate_expression2(f_vec2, exp_vec2):
    c1, c2 = f_vec2
    exp1, exp2 = exp_vec2
    expression = smt.Times(
        smt.Times(smt.Real(c1), smt.Pow(x, smt.Real(exp1))),
        smt.Plus(smt.Real(c2), smt.Pow(x, smt.Real(exp2))),
    )
    polynomial = PolynomialParser([x], env).parse(expression)
    assert equivalent_expressions(
        powsimplifier.simplify(expression), polynomial.to_pysmt()
    )

    if c1 == 0:
        assert polynomial.degree == 0
        assert polynomial.is_zero

    if exp2 == 0:
        assert same_numerical_constant(
            polynomial.monomials.get((exp1,), 0), c1 * (c2 + 1)
        )

        assert polynomial.degree == (exp1 * int((c1 * (c2 + 1)) != 0))
    else:
        assert same_numerical_constant(polynomial.monomials.get((exp1,), 0), (c1 * c2))
        assert same_numerical_constant(polynomial.monomials.get((exp1 + exp2,), 0), c1)

        assert polynomial.degree == (exp1 + exp2) * int(c1 != 0)


def test_multivariate_expression(f_vec2, exp_vec2):
    c1, c2 = f_vec2
    exp1, exp2 = exp_vec2
    expression = smt.Plus(
        smt.Times(smt.Real(c1), smt.Pow(x, smt.Real(exp1))),
        smt.Times(smt.Real(c2), smt.Pow(y, smt.Real(exp2))),
    )
    polynomial = PolynomialParser([x, y], env).parse(expression)
    assert equivalent_expressions(
        powsimplifier.simplify(expression), polynomial.to_pysmt()
    )
    if exp1 != 0:
        assert same_numerical_constant(polynomial.monomials.get((exp1, 0), 0), c1)
    if exp2 != 0:
        assert same_numerical_constant(polynomial.monomials.get((0, exp2), 0), c2)
    assert same_numerical_constant(
        polynomial.monomials.get((0, 0), 0),
        int(exp1 == 0) * c1 + int(exp2 == 0) * c2,
    )
    assert polynomial.degree == max(exp1 * int(c1 != 0), exp2 * int(c2 != 0))
