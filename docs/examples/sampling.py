from itertools import product
from pysmt.shortcuts import *
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np

from wmpy.core import Polynomial
from wmpy.solvers import WMSampler

N_SAMPLES = 5000
DPU = 100

smt_env = get_env()

x = Symbol("x", REAL)
y = Symbol("y", REAL)

formula = support = And(
    LE(Real(0), x),
    LE(Real(0), y),
    LE(x, Real(1)),
    LE(y, Real(1)),
    Or(
        GE(y, Plus(x, Real(1 / 4))),
        LE(y, Plus(x, Real(-1 / 4))),
        LE(Plus(x, y), Real(3 / 4)),
        GE(Plus(x, y), Real(5 / 4)),
    ),
)

weights = Plus(x, y)

sampler = WMSampler(formula, weights, [x, y])
samples = sampler.sample(N_SAMPLES, max_iterations=10)

ax1 = plt.subplot(1, 2, 1)
ax1.scatter(samples[:,0], samples[:,1], marker='x', alpha=0.5)

ax2 = plt.subplot(1, 2, 2)
#uni = np.array(list(product(np.arange(0, 1, 1/DPU), repeat=2)))
sampler = WMSampler(formula, Real(1), [x, y])
uni = sampler.sample(N_SAMPLES*10, max_iterations=10)
f = Polynomial(weights, [x, y], smt_env).to_numpy()
ax2.scatter(uni[:,0], uni[:,1], marker='x', alpha=1.0, color=cm.viridis(f(uni)))

plt.show()
