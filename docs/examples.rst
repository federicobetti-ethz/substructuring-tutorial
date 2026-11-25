Examples
========

Both scripts are run from the repository root, after ``poetry install``. They
share one specimen: a :math:`2\,\mathrm{m}` by :math:`0.35\,\mathrm{m}`
rectangle, thickness :math:`10\,\mathrm{mm}`, with

.. math::

   E = 210\,\mathrm{GPa}, \quad \nu = 0.3, \quad \rho = 7800\,\mathrm{kg\,m^{-3}}.

The mesh is :math:`40 \times 10` quadratic quadrilaterals. The wall is the edge
:math:`x = 0`. A chain can be cut only where element edges meet, so the number
of elements along the beam has to be divisible by the number of pieces.

.. code-block:: bash

   poetry run python scripts/cantilever.py
   poetry run python scripts/modes.py

Figures are written to ``figures/``, which is not part of the repository.

Tip history
-----------

``scripts/cantilever.py`` builds the unreduced cantilever and the same beam
cut into 2 pieces. Every piece is reduced with 2, 4, 6, 8, 12, 16, 24, and
32 elastic modes.
Each piece is a :class:`~substructuring.substructure.Substructure`.
``reduce("craig-bampton")`` and ``reduce("rubin")`` replace its interior.
Rubin uses ``n_rigid = 0`` on the root and ``n_rigid = 3`` on the others.
:class:`~substructuring.assembly.GlobalModel` couples the pieces with the
localization :math:`L`, forming :math:`L^\mathsf{T} K L` and
:math:`L^\mathsf{T} M L`. Guyan reduction is the same call with
``"guyan"``.

Before the sweep, the script checks that a Craig–Bampton chain with every
fixed-interface mode recovers the natural frequencies of a coarse unreduced
beam, for two pieces and for three.

The plotted load acts at the upper free corner, with the same value in
:math:`x` and in :math:`y`. It grows as a raised cosine over
:math:`0.08` of the first natural period and is then held at :math:`10\,\mathrm{kN}`:

.. math::

   p(t) =
   \begin{cases}
   \tfrac{1}{2}\bigl(1 - \cos(\pi t / T)\bigr) & 0 \le t \le T, \\
   1 & t \ge T.
   \end{cases}

The response is the exact modal solution of
:math:`\ddot s + 2\zeta\omega \dot s + \omega^2 s = \varphi^\mathsf{T} f\, p(t)`
for every mode of the model being integrated, with :math:`\zeta = 0.01`.

Component modes
---------------

``scripts/modes.py`` keeps the beam in two pieces and draws three elastic modes
of each, together with the interface shapes. Craig–Bampton shows the static
response to a horizontal translation, a vertical translation, and a rotation
of the whole cut, then the first fixed-interface modes. Rubin shows, 
on the free substructure, the three rigid motions, then the residual
flexibility for the same three motions of the whole cut, then the first
free-interface modes. The clamped substructure has no rigid modes; its column is
padded so that the residual flexibility lines up with the free substructure.

Implementation details
----------------------

The full model matrices are given by scikit-fem. To be coherent with what the majority of commercial software does,
the reductions in this package are assembled in primal form. Should there be an interest in the dual form, please let us know.