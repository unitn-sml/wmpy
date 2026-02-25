from itertools import product
import numpy as np
import pysmt.shortcuts as smt
import pytest

from wmpy.optimization import CvxpyOptimizer
from wmpy.core import PolynomialParser, Polytope

EPSILON = 1e-20


def close_enough(v1, v2):
    # close dimension-wise or L2 distance-wise
    return np.isclose(v1, v2).all()  # or np.isclose(np.linalg.norm(v1 - v2), 0.0)


@pytest.mark.parametrize("N", range(2, 20, 2))
@pytest.mark.parametrize("logside", [i for i in range(-10, 10, 2)])
def test_hyperrectangle(N, logside):

    epsilon = 10**logside

    env = smt.get_env()
    domain = []
    clauses = []
    for i in range(N):
        var = smt.Symbol(f"x{i}", smt.REAL)
        bounds = [
            smt.LE(smt.Real(float(-epsilon)), var),
            smt.LE(var, smt.Real(float(epsilon))),
        ]
        domain.append(var)
        clauses.extend(bounds)

    polytope = Polytope(clauses, PolynomialParser(domain, env))

    optimizer = CvxpyOptimizer()
    lower, upper, optimal = optimizer.compute_inner_box(polytope, epsilon=EPSILON)

    assert (lower <= upper).all()

    gt_lower = -np.ones(N) * epsilon
    gt_upper = np.ones(N) * epsilon
    assert (
        close_enough(gt_lower, lower) or not optimal
    ), f"Lower - expected: {gt_lower} --- got: {lower}"
    assert (
        close_enough(gt_upper, upper) or not optimal
    ), f"Upper - expected: {gt_upper} --- got: {upper}"


@pytest.mark.parametrize("N", range(2, 10, 1))
@pytest.mark.parametrize("logside", [i for i in range(-10, 10, 2)])
def test_oblique(N, logside):

    epsilon = 10**logside
    gt_lower = -np.ones(N) * epsilon
    gt_upper = np.ones(N) * epsilon

    env = smt.get_env()
    domain = []
    clauses = []
    for i in range(N):
        var = smt.Symbol(f"x{i}", smt.REAL)
        bounds = [
            smt.LE(smt.Real(float(-epsilon)), var),
            smt.LE(var, smt.Real(float(epsilon))),
        ]
        domain.append(var)
        clauses.extend(bounds)

    for coeffs in product([-1, 1], repeat=N):
        clauses.append(
            smt.LE(
                smt.Plus(
                    *[smt.Times(smt.Real(coeffs[i]), domain[i]) for i in range(N)]
                ),
                smt.Real(epsilon),
            )
        )

    polytope = Polytope(clauses, PolynomialParser(domain, env))

    optimizer = CvxpyOptimizer()
    lower, upper, optimal = optimizer.compute_inner_box(polytope, epsilon=EPSILON)

    assert (lower <= upper).all()

    gt_lower = -np.ones(N) * epsilon / N
    gt_upper = np.ones(N) * epsilon / N
    assert (
        close_enough(gt_lower, lower) or not optimal
    ), f"Lower - expected: {gt_lower} --- got: {lower}"
    assert (
        close_enough(gt_upper, upper) or not optimal
    ), f"Upper - expected: {gt_upper} --- got: {upper}"
