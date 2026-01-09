"""The wmpy.optimization submodule handles anything related solving convex optimization with polynomial objective functions.

It exposes the following samplers:
- CvxpyOptimizer: an optimizer based on cvxpy
- ScipyOptimizer: an optimizer based on scipy
"""

from .cvxpyoptimizer import CvxpyOptimizer
from .scipyoptimizer import ScipyOptimizer
