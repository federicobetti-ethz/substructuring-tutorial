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
cut into 2 pieces. Every piece is reduced with 2, 8, 16, and 32 elastic modes.
:func:`substructuring.craig_bampton.craig_bampton` is applied to each piece.
:func:`substructuring.rubin.rubin` uses ``n_rigid = 0`` on the root and
``n_rigid = 3`` on the others. The pieces are assembled with
:func:`substructuring.coupling.couple`.

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
   1 & t > T.
   \end{cases}

The response is the exact modal solution of
:math:`\ddot s + 2\zeta\omega \dot s + \omega^2 s = \varphi^\mathsf{T} f\, p(t)`
for every mode of the model being integrated, with :math:`\zeta = 0.01`. The
window covers six periods of the unreduced fundamental. ``figures/tip_history.png``
has one column per elastic-mode count and shows the horizontal and the vertical
displacement of that corner for the full model, Rubin, and Craig–Bampton.
``figures/tip_norm_error.png`` reports that same comparison as a percentage:
the relative :math:`L^2` error of the tip-norm history :math:`\|u(t)\|`.

Component modes
---------------

``scripts/modes.py`` keeps the beam in two pieces and draws three elastic modes
of each, together with one interface shape. Craig–Bampton shows the vertical
constraint mode through the middle of the cut and the first fixed-interface
modes. Rubin shows, on the free piece, the three rigid motions, the vertical
attachment mode, and the first free-interface modes. The clamped piece has no
rigid modes; its column is padded so that the attachment mode lines up with
the free piece.

Each panel uses the outline of the whole beam. The active piece is deformed and
the other piece is left in place. The output is ``figures/craig_bampton_modes.png``
and ``figures/rubin_modes.png``.
