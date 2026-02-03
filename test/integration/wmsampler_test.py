import pytest
import pysmt.shortcuts as smt
from wmpy.enumeration import SAEnumerator
from wmpy.integration import LattEIntegrator
from wmpy.solvers import WMSampler

SEED = 666
N_VARS = 3
N_SAMPLES = 500

env = smt.get_env()


def test_no_infeasible_sample(enumerator, exact_integrator):
    domain = [smt.Symbol(f"x{i}", smt.REAL) for i in range(N_VARS)]

    chi = smt.And(
        *[smt.And(smt.LE(smt.Real(0), v), smt.LE(v, smt.Real(1))) for v in domain]
    )

    asc = []
    desc = []
    for i in range(N_VARS - 2):
        vi, vj, vk = domain[i], domain[i + 1], domain[i + 2]
        asc.append(smt.Implies(smt.LE(vi, vj), smt.LE(vj, vk)))
        desc.append(smt.Implies(smt.GE(vi, vj), smt.GE(vj, vk)))

    chi = smt.And(chi, smt.Or(smt.And(*asc), smt.And(*desc)))
    sampler = WMSampler(
        enumerator(chi, smt.Real(1), env), domain, exact_integrator(), seed=SEED
    )
    samples = sampler.sample(N_SAMPLES, max_iterations=10)
    print(smt.serialize(chi))
    for s in samples:
        chi_s = smt.And(
            *[smt.Equals(v, smt.Real(float(s[i]))) for i, v in enumerate(domain)]
        )
        assert smt.is_sat(smt.And(chi, chi_s)), f"Unfeasible sample: {s}"


def test_null_density_error(enumerator, exact_integrator):
    env = smt.get_env()
    x = smt.Symbol("x", smt.REAL)
    domain = [x]
    support = smt.And(smt.LE(smt.Real(0), x), smt.LE(x, smt.Real(1)))
    zero_weight = smt.Real(0)
    with pytest.raises(ValueError):
        sampler = WMSampler(
            enumerator(support, zero_weight, env), domain, exact_integrator(), seed=SEED
        )


def test_unsat_support_error(enumerator, exact_integrator):
    env = smt.get_env()
    x = smt.Symbol("x", smt.REAL)
    domain = [x]
    unsat_support = smt.And(smt.LE(smt.Real(0), x), smt.LE(x, smt.Real(-1)))
    weight = smt.Real(1)

    with pytest.raises(ValueError):
        sampler = WMSampler(
            enumerator(unsat_support, weight, env),
            domain,
            exact_integrator(),
            seed=SEED,
        )
