import itertools
import pytest
import pysmt.shortcuts as smt
from wmpy.integration import LattEIntegrator
from wmpy.solvers import IncrementalBounder, WMISolver


@pytest.mark.parametrize("N, B", itertools.product([2, 3], repeat=2))
def test_incremental(enumerator, N, B):
    env = smt.get_env()
    domain = [smt.Symbol(f"x{i}", smt.REAL) for i in range(N)]
    booleans = [smt.Symbol(f"A{i}", smt.BOOL) for i in range(B)]

    clauses = []
    factors = []
    for rvar in domain:
        clauses.extend([smt.LE(smt.Real(0), rvar), smt.LE(rvar, smt.Real(1))])
        factors.append(rvar)

    for i in range(len(domain) - 1):
        bvar = booleans[i % len(booleans)]
        rvar1, rvar2 = domain[i], domain[i + 1]
        clauses.append(smt.Implies(bvar, smt.LE(rvar1, rvar2)))
        factors.append(
            smt.Ite(
                smt.LE(rvar1, smt.Real(1 / 2)),
                smt.Times(rvar2, rvar2, smt.Plus(rvar1, smt.Real(i))),
                smt.Real(1),
            )
        )

    support = smt.And(*clauses)
    weight = smt.Times(*factors)

    enum = enumerator(support, weight, env)
    ground_truth = WMISolver(enum, domain, integrator=LattEIntegrator()).compute(
        smt.Bool(True)
    )["wmi"]

    i = 0
    bounder = IncrementalBounder(enum, domain)
    assert (
        bounder.lower_bound <= ground_truth
    ), f"it {i} LB: {bounder.lower_bound} GT: {ground_truth}"
    assert (
        ground_truth <= bounder.upper_bound
    ), f"it {i} UB: {bounder.upper_bound} GT: {ground_truth}"
    for i in range(1, 10):
        last_bounds = (bounder.lower_bound, bounder.upper_bound)
        bounder.refine()
        assert (
            last_bounds[0] <= bounder.lower_bound
        ), f"it {i} LB: {bounder.lower_bound} prev: {last_bounds[0]}"
        assert (
            last_bounds[1] >= bounder.upper_bound
        ), f"it {i} UB: {bounder.upper_bound} prev: {last_bounds[1]}"
        assert (
            bounder.lower_bound <= ground_truth
        ), f"it {i} LB: {bounder.lower_bound} GT: {ground_truth}"
        assert (
            ground_truth <= bounder.upper_bound
        ), f"it {i} UB: {bounder.upper_bound} GT: {ground_truth}"
