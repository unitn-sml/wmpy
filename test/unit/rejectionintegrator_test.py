import itertools
import numpy as np
import pytest

import pysmt.shortcuts as smt

from wmpy.core import PolynomialParser, Polytope
from wmpy.integration import RejectionIntegrator


@pytest.mark.parametrize(
    "n, xmin, side", itertools.product([2, 3, 4], [-1, 0, 100], [1, 10, 100])
)
def test_integrate_axisaligned_constant(n, xmin, side):
    env = smt.get_env()
    variables = [smt.Symbol(f"x{i}", smt.REAL) for i in range(n)]
    parser = PolynomialParser(variables, env)

    inequalities = []
    for xi in variables:
        inequalities.extend(
            [smt.LE(smt.Real(xmin), xi), smt.LE(xi, smt.Real(xmin + side))]
        )

    volume = side ** (n + 1)
    polynomial = parser.parse(smt.Real(float(side)))
    polytope = Polytope(inequalities, parser)
    result = RejectionIntegrator().integrate(polytope, polynomial)
    assert np.isclose(result, volume), f"Expected {side} ^ {n} = {volume}, got {result}"


@pytest.mark.parametrize("n, batch_size", itertools.product([2, 3, 4], repeat=2))
def test_integrate__axisaligned_constant_batch(n, batch_size):
    np.random.seed(10 * n + batch_size)
    env = smt.get_env()
    variables = [smt.Symbol(f"x{i}", smt.REAL) for i in range(n)]
    parser = PolynomialParser(variables, env)

    volume = []
    integrals = []
    for npoly in range(batch_size):
        volume.append(1)
        inequalities = []
        for xi in variables:
            xmin, xmax = sorted(np.random.random(2))
            inequalities.extend(
                [smt.LE(smt.Real(float(xmin)), xi), smt.LE(xi, smt.Real(float(xmax)))]
            )
            volume[-1] *= xmax - xmin

        weight = np.random.random()
        volume[-1] *= weight
        polynomial = parser.parse(smt.Real(float(weight)))
        integrals.append((Polytope(inequalities, parser), polynomial))

    result = RejectionIntegrator().integrate_batch(integrals)
    assert np.isclose(
        result, volume
    ).all(), f"Expected {side} ^ {n} = {volume}, got {result}"
