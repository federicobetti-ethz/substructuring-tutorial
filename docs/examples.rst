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
shows the horizontal and the vertical displacement of that corner for the full
model, static condensation, Rubin, and Craig–Bampton, with 16 elastic modes in
each piece. ``figures/tip_error.png`` sweeps the elastic-mode count and
reports the relative :math:`L^2` error of the horizontal and the vertical
tip displacement.
``figures/tip_norm_error.png`` is that same sweep for the tip norm alone.
``figures/frequency_convergence.png`` follows each assembled frequency up to
1700 Hz from one elastic-mode count to the next. A point is the relative
change from the previous refinement.

The same histories include a chain reduced by
:func:`substructuring.static_condensation.static_condensation`. That basis
keeps only the interface, so the dotted curve is independent of the elastic-mode
count. ``figures/tip_condensation.png`` shows that comparison alone, for the
horizontal and the vertical tip displacement.
``figures/tip.png`` marks the measured corner in red. The script prints the
relative :math:`L^2` error beside the mode sweep.

Component modes
---------------

``scripts/modes.py`` keeps the beam in two pieces and draws three elastic modes
of each, together with the interface shapes. Craig–Bampton shows the static
response to a horizontal translation, a vertical translation, and a rotation
of the whole cut, then the first fixed-interface modes. Rubin shows, on the free piece, the three rigid motions, then the residual
flexibility for the same three motions of the whole cut, then the first
free-interface modes. The clamped piece has no rigid modes; its column is
padded so that the residual flexibility lines up with the free piece.

Each mode is one figure, with both substructures in it. A piece that does not
have that mode stays in place as the gray outline. The files are
``figures/craig_bampton/`` and ``figures/rubin/``.

``figures/vibration/`` holds the first vibration modes of the unreduced
cantilever, one figure per mode. ``figures/model.png`` is the mesh itself.
``figures/primal.png`` and ``figures/dual.png`` each start with the two
substructures apart. The primal figure then joins them, so the cut carries one
displacement unknown. The dual figure ties each matching pair of interface
nodes with a force unknown. ``figures/condensation.png`` marks the nodes of
each piece: filled nodes on the cut are retained, open nodes are condensed,
and the squares on the wall are prescribed. The lower panel keeps only the
retained nodes. The reductions in this package are assembled in
the primal form.
