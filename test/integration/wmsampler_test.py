
import pysmt.shortcuts as smt

from wmpy.solvers import WMSampler

SEED=666
N_VARS = 3
N_SAMPLES = 500

env = smt.get_env()

def test_no_infeasible_sample(enumerator, exact_integrator):
    domain = [smt.Symbol(f"x{i}", smt.REAL) for i in range(N_VARS)]
    
    chi = smt.And(*[smt.And(smt.LE(smt.Real(0), v), smt.LE(v, smt.Real(1)))
                    for v in domain])
    
    for i in range(N_VARS-2):
        vi, vj, vk = domain[i], domain[i+1], domain[i+2]
        chi = smt.And(chi, smt.Implies(smt.LE(vi, vj), smt.LE(vj, vk)))
                      
    sampler = WMSampler(enumerator(chi, smt.Real(1), env), domain, exact_integrator(), seed=SEED)
    samples = sampler.sample(N_SAMPLES, max_iterations=10)
    print(smt.serialize(chi))    
    for s in samples:
        chi_s = smt.And(*[smt.Equals(v, smt.Real(float(s[i]))) for i, v in enumerate(domain)])
        assert smt.is_sat(smt.And(chi, chi_s)), f"Unfeasible sample: {s}"
