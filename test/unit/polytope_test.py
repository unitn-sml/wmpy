from itertools import product
import numpy as np
import pytest

import pysmt.shortcuts as smt

from wmpy.core.polytope import Polytope


def test_polytope_nonlinear():
    env = smt.get_env()
    x = smt.Symbol("x", smt.REAL)
    y = smt.Symbol("y", smt.REAL)
    inequalities = [
        smt.LE(x, smt.Real(0)),
        smt.GE(smt.Pow(y, smt.Real(3)), smt.Real(0)),
    ]
    with pytest.raises(AssertionError):
        polytope = Polytope(inequalities, [x, y], env=env)


def test_polytope_unknown_variable():
    env = smt.get_env()
    x = smt.Symbol("x", smt.REAL)
    y = smt.Symbol("y", smt.REAL)
    z = smt.Symbol("z", smt.REAL)
    inequalities = [
        smt.LE(x, smt.Real(0)),
        smt.GE(z, smt.Real(1)),
    ]
    with pytest.raises(AssertionError):
        polytope = Polytope(inequalities, [x, y], env=env)


@pytest.fixture(params=product(range(1, 5), repeat=2))
def Ab(request):
    M, N = request.param
    np.random.seed(10 * N + M)
    A = np.random.random((M, N))
    b = np.random.random((M,))
    return A, b


def test_polytope_conversion(Ab):
    env = smt.get_env()
    A, b = Ab
    M, N = A.shape
    vx = [smt.Symbol(f"x{i}", smt.REAL) for i in range(N)]
    inequalities = []
    for i in range(M):
        Asmt = smt.Plus([smt.Times(smt.Real(float(A[i][j])), vx[j]) for j in range(N)])
        bsmt = smt.Real(float(b[i]))
        relsmt = smt.LE if i % 2 == 0 else smt.LT
        inequalities.append(relsmt(Asmt, bsmt))

    polytope = Polytope(inequalities, vx, env=env)
    Aconv, bconv, sconv = polytope.to_numpy()
    assert (Aconv == A).all(), f"numpy conversion error\nA:\n{A}\nA':\n{Aconv}"
    assert (bconv == b).all(), f"numpy conversion error\nb:\n{b}\nb':\n{bconv}"
    assert (
        sconv == np.array([i % 2 for i in range(M)])
    ).all(), f"numpy conversion error\ns:\n{sconv}\nstrictness vector non-matching"

    f = smt.And(*inequalities)
    fconv = polytope.to_pysmt()
    assert not smt.is_sat(
        smt.Not(smt.Iff(f, fconv))
    ), "pysmt conversion error\nf:\n{smt.serialize(f)}\nf':{smt.serialize(fconv)}"


env = smt.get_env()
x, y = smt.Symbol("x", smt.REAL), smt.Symbol("y", smt.REAL)
box = [
    smt.LE(smt.Real(0), x),
    smt.LE(x, smt.Real(1)),
    smt.LE(smt.Real(0), y),
    smt.LE(y, smt.Real(1)),
]

h1 = smt.LE(smt.Plus(x, y), smt.Real(1))
h2 = smt.LE(y, x)
h3 = smt.LE(smt.Real(1 / 2), x)


def test_polytope_outer_box1():
    lower, upper = Polytope(box, [x, y], env=env).outer_box
    expl, expu = np.array([0, 0]), np.array([1, 1])
    assert np.isclose(lower, expl).all() and np.isclose(upper, expu).all()


def test_polytope_outer_box2():
    lower, upper = Polytope(box + [h1], [x, y], env=env).outer_box
    expl, expu = np.array([0, 0]), np.array([1, 1])
    assert np.isclose(lower, expl).all() and np.isclose(upper, expu).all()


def test_polytope_outer_box3():
    lower, upper = Polytope(box + [h2], [x, y], env=env).outer_box
    expl, expu = np.array([0, 0]), np.array([1, 1])
    assert np.isclose(lower, expl).all() and np.isclose(upper, expu).all()


def test_polytope_outer_box4():
    lower, upper = Polytope(box + [h1, h2], [x, y], env=env).outer_box
    expl, expu = np.array([0, 0]), np.array([1, 1 / 2])
    assert np.isclose(lower, expl).all() and np.isclose(upper, expu).all()


def test_polytope_outer_box5():
    lower, upper = Polytope(box + [h1, h2, h3], [x, y], env=env).outer_box
    expl, expu = np.array([1 / 2, 0]), np.array([1, 1 / 2])
    assert np.isclose(lower, expl).all() and np.isclose(upper, expu).all()


def test_polytope_inner_box1():
    lower, upper = Polytope(box, [x, y], env=env).inner_box
    expl, expu = np.array([0, 0]), np.array([1, 1])
    assert np.isclose(lower, expl).all() and np.isclose(upper, expu).all()


def test_polytope_inner_box2():
    lower, upper = Polytope(box + [h1], [x, y], env=env).inner_box
    expl, expu = np.array([0, 0]), np.array([1 / 2, 1 / 2])
    assert np.isclose(lower, expl).all() and np.isclose(upper, expu).all()


def test_polytope_inner_box3():
    lower, upper = Polytope(box + [h2], [x, y], env=env).inner_box
    expl, expu = np.array([1 / 2, 0]), np.array([1, 1 / 2])
    assert np.isclose(lower, expl).all() and np.isclose(upper, expu).all()


def test_polytope_inner_box4():
    lower, upper = Polytope(box + [h1, h2], [x, y], env=env).inner_box
    expl, expu = np.array([1 / 4, 0]), np.array([3 / 4, 1 / 4])
    assert np.isclose(lower, expl).all() and np.isclose(upper, expu).all()


def test_polytope_inner_box5():
    lower, upper = Polytope(box + [h1, h2, h3], [x, y], env=env).inner_box
    expl, expu = np.array([1 / 2, 0]), np.array([3 / 4, 1 / 4])
    assert np.isclose(lower, expl).all() and np.isclose(upper, expu).all()
