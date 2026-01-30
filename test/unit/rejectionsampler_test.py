import numpy as np
import pytest

import pysmt.shortcuts as smt

from wmpy.core import Polynomial, Polytope
from wmpy.sampling import RejectionSampler

SEED = 666
N_SAMPLES = 777


@pytest.mark.parametrize("n", [2, 3, 4])
def test_no_infeasible_sample(n):
    env = smt.get_env()
    inequalities = []
    variables = [smt.Symbol(f"x{i}", smt.REAL) for i in range(n)]
    for xi in variables:
        inequalities.extend([smt.LE(smt.Real(0), xi), smt.LE(xi, smt.Real(1))])

    for i in range(len(variables) - 1):
        inequalities.append(smt.LE(variables[i], variables[i + 1]))

    polynomial = Polynomial(smt.Real(1), variables, env)
    polytope = Polytope(inequalities, variables, env)
    sample = RejectionSampler(polytope, polynomial, SEED).sample(N_SAMPLES)
    for s in sample:
        chi_s = smt.And(
            *[smt.Equals(v, smt.Real(float(s[i]))) for i, v in enumerate(variables)]
        )
        assert smt.is_sat(smt.And(*inequalities, chi_s)), f"Unfeasible sample: {s}"


def test_null_density_error():
    env = smt.get_env()
    x = smt.Symbol("x", smt.REAL)
    variables = [x]
    inequalities = [smt.LE(smt.Real(0), x), smt.LE(x, smt.Real(1))]
    polynomial = Polynomial(smt.Real(0), variables, env)
    polytope = Polytope(inequalities, variables, env)
    with pytest.raises(ValueError):
        sample = RejectionSampler(polytope, polynomial, SEED).sample(N_SAMPLES)
