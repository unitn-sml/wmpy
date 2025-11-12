"""The wmpy.sampling submodule handles anything related to sampling over weighted convex polytopes.

It exposes the following samplers:

- RejectionSampler: an approximate integrator based on rejection sampling
"""

from .rejection import RejectionSampler
