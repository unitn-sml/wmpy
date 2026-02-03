import pysmt.shortcuts as smt
import pytest
from pysmt.typing import REAL

from wmpy.core import PolynomialParser
from wmpy.core.inequality import Inequality

x = smt.Symbol("X", REAL)
y = smt.Symbol("Y", REAL)
z = smt.Symbol("Z", REAL)
v1 = -3
v2 = 5
v3 = 2
r1 = smt.Real(v1)
r2 = smt.Real(v2)
r3 = smt.Real(v3)
env = smt.get_env()


def equivalent_expressions(original, converted):
    return not smt.is_sat(smt.Not(smt.Iff(original, converted)))


def test_ineq_degree_zero():
    parser = PolynomialParser([x], env)
    with pytest.raises(ValueError):
        _ = Inequality(smt.LE(smt.Real(0), smt.Real(1)), parser)


def test_ineq_degree_more_than_one():
    parser = PolynomialParser([x], env)
    expression = smt.LE(smt.Pow(x, smt.Real(2)), r1)
    with pytest.raises(ValueError):
        _ = Inequality(expression, parser)


def test_not_an_ineq():
    parser = PolynomialParser([x], env)
    expression = smt.Equals(smt.Pow(x, smt.Real(2)), r1)
    with pytest.raises(ValueError):
        _ = Inequality(expression, parser)


@pytest.mark.parametrize(
    "expression",
    [
        smt.LE(x, smt.Real(5)),
        smt.LE(smt.Real(0), smt.Plus(x, smt.Real(5))),
        smt.LE(smt.Plus(x, smt.Real(5)), smt.Plus(x, x)),
        smt.LE(smt.Times(x, smt.Real(3.5)), x),
        smt.LE(smt.Times(x, smt.Real(3.5)), smt.Plus(smt.Real(5), x)),
    ],
)
def test_ineq_univariate(expression):
    parser = PolynomialParser([x], env)
    ineq = Inequality(expression, parser)
    assert equivalent_expressions(expression, ineq.to_pysmt())
    A, b = ineq.to_numpy()
    from_numpy = smt.LE(smt.Times(smt.Real(float(A[0])), x), smt.Real(float(b)))
    assert equivalent_expressions(expression, from_numpy)


@pytest.mark.parametrize(
    "expression",
    [
        smt.LE(x, y),
        smt.LE(smt.Real(0), smt.Plus(x, y)),
        smt.LE(smt.Plus(x, smt.Real(5)), y),
        smt.LE(smt.Times(x, smt.Real(3.5)), y),
        smt.LE(smt.Times(x, smt.Real(3.5)), smt.Plus(smt.Real(5), y)),
    ],
)
def test_ineq_no_bivariate(expression):
    parser = PolynomialParser([x, y], env)
    ineq = Inequality(expression, parser)
    A, b = ineq.to_numpy()
    from_numpy = smt.LE(
        smt.Plus(
            smt.Times(smt.Real(float(A[0])), x), smt.Times(smt.Real(float(A[1])), y)
        ),
        smt.Real(float(b)),
    )
    assert equivalent_expressions
