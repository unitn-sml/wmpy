..
   Note: Items in this toctree form the top-level navigation. See `api.rst` for the `autosummary` directive, and for why `api.rst` isn't called directly.

.. toctree::
   :hidden:
   :maxdepth: 0

   Home page <self>
   Theory <theory>
   Get started <getstarted>
   Jupyter tutorials <tutorials>
   API reference <_autosummary/wmpy>
   Development <development>

.. autosummary::
   :hidden:
   :recursive:

   wmpy


Welcome to the WM*PY documentation
==================================

.. centered::
   ``wmpy`` :math:`=` reasoning over **weighted algebraic** and
   **logical constraints** in ``python3``

.. math::

   P(x \ge y \:|\: y / 2 \le \pi) = ?? \quad\quad\quad\quad \text{ (marginals)}

   P(x + y > 3/4) \le 0.01 ??  \quad\quad\quad\quad \text{ (bounds)}

   argmax_y \: P(x, y) = ?? \quad\quad \text{ (optimization)}

   x \sim P(x \: | \: x \ge y) \quad\quad\quad\quad \text{ (sampling)}

.. image:: images/intro-cropped.png
   :scale: 30 %
   :align: center

.. centered::
   ... ``pip install wmpy`` !

**WM\*** (weighted model \*) denotes any problem that is defined on the
models (solutions) of an SMT-LRA formula, i.e. any combination of
logical and linear constraints over Boolean and continuous variables.

**WM\*PY** (``wmpy``) is a modular library for solving quantitative reasoning tasks over logical and algebraic constraints.

This library:

* Facilitates the integration of state-of-the-art WM* technology into your project
* Provides a flexible framework for the development of novel solvers



**Unfamiliar with WMI or SMT?** :ref:`Read <theory>` our theory primer first.

**Eager to code?** :ref:`Get started <getstarted>` now!

**Curious about advanced use cases?** :ref:`Check out <tutorials>` our Jupiter notebooks.

**Want to be part of** ``wmpy`` **?** :ref:`Learn <development>` how to contribute.
