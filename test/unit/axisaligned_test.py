import numpy as np
import pytest

import pysmt.shortcuts as smt

from wmpy.core import Polynomial, Polytope
from wmpy.integration import AxisAlignedWrapper, LattEIntegrator

env = smt.get_env()
np.random.seed(666)

x = smt.Symbol("x", smt.REAL)
y = smt.Symbol("y", smt.REAL)
variables = [x, y]


class DummyIntegrator:

    def integrate(self, polytope, polynomial):
        raise Exception("AxisAlignedWrapper didn't catch this.")


def hypercube(variables, sides, offsets):
    inequalities = []
    for i, var in enumerate(variables):
        inequalities.extend(
            [
                smt.LE(smt.Real(float(offsets[i])), var),
                smt.LE(var, smt.Real(float(offsets[i] + sides[i]))),
            ]
        )

    return inequalities


def test_integrate(f_vec2, exp_vec2):

    aa_integrator = AxisAlignedWrapper(DummyIntegrator())
    gt_integrator = LattEIntegrator()

    sides = [1, 10, 100]
    offsets = [-10, 0, 100]

    inequalities = hypercube(variables, sides, offsets)
    aa_polytope = Polytope(inequalities, variables, env)

    exp1, exp2 = exp_vec2
    c1, c2 = f_vec2
    expression = smt.Plus(
        smt.Times(
            smt.Real(c1), smt.Pow(x, smt.Real(exp1 + 1)), smt.Pow(y, smt.Real(exp2 + 1))
        ),
        smt.Times(smt.Real(c2), smt.Pow(x, smt.Real(exp1))),
        smt.Times(smt.Real(c1 + c2), smt.Pow(x, smt.Real(exp2))),
    )

    integrand = Polynomial(expression, variables, env)

    result = aa_integrator.integrate(aa_polytope, integrand)
    gt = gt_integrator.integrate(aa_polytope, integrand)
    assert np.isclose(result, gt)

    oblique = smt.LE(
        smt.Plus(*[var for var in variables]),
        smt.Real(float(np.sum([sides, offsets]))),
    )
    oblique_polytope = Polytope(inequalities + [oblique], variables, env)
    with pytest.raises(Exception):
        result = aa_integrator.integrate(oblique_polytope, integrand)


@pytest.mark.parametrize(
    "sides, offsets",
    [
        ([10, 10], [0, 0]),
        ([10, 10], [-10, 6]),
        ([3, 7], [-10, 6]),
    ],
)
def test_integrate_batch(sides, offsets):

    aa_integrator = AxisAlignedWrapper(DummyIntegrator())

    inequalities1 = hypercube(variables, sides, offsets)
    polytope1 = Polytope(inequalities1, variables, env)

    sides2 = sides * np.array([3, 10])
    offsets2 = offsets + np.array(sides)
    inequalities2 = hypercube(variables, sides2, offsets2)
    polytope2 = Polytope(inequalities2, variables, env)

    k1, k2 = 3.33, 15
    polynomial1 = Polynomial(smt.Real(k1), variables, env)
    polynomial2 = Polynomial(smt.Real(k2), variables, env)

    batch = [
        (polytope1, polynomial1),
        (polytope1, polynomial2),
        (polytope2, polynomial1),
        (polytope2, polynomial2),
    ]

    expected_result = np.array(
        [
            np.prod(sides) * k1,
            np.prod(sides) * k2,
            np.prod(sides2) * k1,
            np.prod(sides2) * k2,
        ]
    )

    result = aa_integrator.integrate_batch(batch)
    assert np.isclose(result, expected_result).all()
