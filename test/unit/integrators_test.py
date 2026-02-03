import pytest
import pysmt.shortcuts as smt

from wmpy.core import PolynomialParser, Polytope
from wmpy.integration import LattEIntegrator, RejectionIntegrator

env = smt.get_env()
x = smt.Symbol("x", smt.REAL)
y = smt.Symbol("y", smt.REAL)
variables = [x, y]
parser = PolynomialParser(variables, env)


@pytest.fixture(params=[LattEIntegrator(), RejectionIntegrator()])
def integrator(request):
    return request.param


def test_zero_integrand(integrator):

    bbox = [
        smt.LE(smt.Real(-1), x),
        smt.LE(x, smt.Real(1)),
        smt.LE(smt.Real(-1), y),
        smt.LE(y, smt.Real(1)),
    ]
    polytope = Polytope(bbox, parser)
    polynomial = parser.parse(smt.Real(0))

    result = integrator.integrate(polytope, polynomial)
    assert (
        result == 0
    ), f"Expected 0.0, got {result} for {integrator.__class__.__name__}"


def test_zero_volume(integrator):

    zero_volume_bbox = [
        smt.LE(smt.Real(1), x),
        smt.LE(x, smt.Real(1)),
        smt.LE(smt.Real(1), y),
        smt.LE(y, smt.Real(1)),
    ]

    polytope = Polytope(zero_volume_bbox, parser)
    polynomial = parser.parse(smt.Real(1))

    result = integrator.integrate(polytope, polynomial)
    assert (
        result == 0
    ), f"Expected zero volume, got {result} with {integrator.__class__.__name__}, polytope: {polytope}"
