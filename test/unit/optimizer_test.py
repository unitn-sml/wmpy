import numpy as np
import pysmt.shortcuts as smt
import pytest

from wmpy.core import PolynomialParser, Polytope
from wmpy.optimization import ScipyOptimizer


@pytest.fixture(params=[ScipyOptimizer()])
def optimizer(request):
    return request.param


env = smt.get_env()
LOWER, UPPER = 1, 3
HALF = (UPPER + LOWER) / 2
x, y = smt.Symbol("x", smt.REAL), smt.Symbol("y", smt.REAL)
variables = [x, y]
bbox = [
    smt.LE(smt.Real(LOWER), x),
    smt.LE(x, smt.Real(UPPER)),
    smt.LE(smt.Real(LOWER), y),
    smt.LE(y, smt.Real(UPPER)),
]

m = lambda p1, p2: (p2[1] - p1[1]) / (p2[0] - p1[0])
q = lambda p1, p2: p1[1] - (m(p1, p2) * p1[0])


p1 = (HALF, LOWER)
p2 = (LOWER, HALF)
p3 = (UPPER, HALF)
p4 = (HALF, UPPER)

h1 = smt.LE(smt.Plus(smt.Times(smt.Real(m(p1, p2)), x), smt.Real(q(p1, p2))), y)
h2 = smt.LE(y, smt.Plus(smt.Times(smt.Real(m(p3, p4)), x), smt.Real(q(p3, p4))))
h3 = smt.LE(smt.Plus(smt.Times(smt.Real(m(p1, p3)), x), smt.Real(q(p1, p3))), y)
h4 = smt.LE(y, smt.Plus(smt.Times(smt.Real(m(p2, p4)), x), smt.Real(q(p2, p4))))

parser = PolynomialParser(variables, env)
p_univ = Polytope(bbox, parser)
p_oblique = Polytope([h1, h2, h3, h4], parser)


f_x = parser.parse(x)
f_y = parser.parse(y)
f_lin1 = parser.parse(smt.Plus(x, y))


@pytest.mark.parametrize(
    "polytope, polynomial, expected_argmin, expected_argmax",
    [
        (p_univ, f_lin1, np.array([LOWER, LOWER]), np.array([UPPER, UPPER])),
        (p_oblique, f_x, np.array([LOWER, HALF]), np.array([UPPER, HALF])),
        (p_oblique, f_y, np.array([HALF, LOWER]), np.array([HALF, UPPER])),
    ],
)
def test_optimizer(optimizer, polytope, polynomial, expected_argmin, expected_argmax):

    argmax = optimizer.optimize(polytope, polynomial, maximize=True)
    argmin = optimizer.optimize(polytope, polynomial, maximize=False)

    assert np.isclose(
        expected_argmax, argmax
    ).all(), f"Expected argmax {expected_argmax}, got {argmax}"

    assert np.isclose(
        expected_argmin, argmin
    ).all(), f"Expected argmin {expected_argmin}, got {argmin}"
