"""The wmpy.core submodule contains task-agnostic classes and utilities.

It exposes:

- AssignmentConverter: a class that facilitates the conversion of satisfying TAs into polytope, polynomial pairs
- Polynomial: the internal class for handling polynomials
- PolynomialParser: the polynomial parser class
- Polytope: the internal class for handling convex polytopes
"""

from .polynomial import Polynomial, PolynomialParser
from .polytope import Polytope
from .assignmentconverter import AssignmentConverter
