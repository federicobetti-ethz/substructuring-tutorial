Theory
======

A component is one piece of the mesh, with its own stiffness :math:`K` and
mass :math:`M`. Component-mode synthesis replaces the unknowns inside that
piece by a few generalised coordinates, while the displacements on each cut
stay as the coordinates that tie the pieces together. The forms below are the
ones implemented in :func:`substructuring.craig_bampton.craig_bampton` and
:func:`substructuring.rubin.rubin`. Both return a basis :math:`R` such that
the physical displacement is :math:`u = R q`, and both reduce the matrices by
the same projection,

.. math::

   \tilde K = R^\mathsf{T} K R, \qquad \tilde M = R^\mathsf{T} M R.

The finite-element component
----------------------------

The specimen is a flat rectangle in plane stress. With Young's modulus
:math:`E` and Poisson's ratio :math:`\nu`,

.. math::

   \mu = \frac{E}{2(1+\nu)}, \qquad
   \lambda = \frac{E \nu}{1-\nu^2}.

Here :math:`\lambda` is the plane-stress modulus, not the Lamé parameter of
three-dimensional elasticity. On a mesh of quadratic quadrilaterals the weak
form, scaled by the thickness :math:`t`, is

.. math::

   a(u, v) = \int_\Omega t \bigl( 2\mu\, \varepsilon(u)\!:\!\varepsilon(v)
   + \lambda\, \operatorname{tr}\varepsilon(u)\, \operatorname{tr}\varepsilon(v) \bigr) \,\mathrm{d}A,

and the mass form is :math:`\int_\Omega t \rho\, u \cdot v \,\mathrm{d}A`.
Clamped degrees of freedom are removed before the reduction, so :math:`K` is
the free stiffness of that piece.

The free unknowns are split into the interface block :math:`u_b` and the
interior block :math:`u_i`. On a cut, the interface block lists every
horizontal unknown in order of increasing :math:`y`, then every vertical
unknown in the same order. An interior piece of a chain stores the left cut
first and the right cut second. Neighbouring pieces can then be coupled entry
by entry. The partitioned equation, still without damping, is

.. math::

   \begin{bmatrix}
   K_{bb} & K_{bi} \\
   K_{ib} & K_{ii}
   \end{bmatrix}
   \begin{Bmatrix} u_b \\ u_i \end{Bmatrix}
   +
   \begin{bmatrix}
   M_{bb} & M_{bi} \\
   M_{ib} & M_{ii}
   \end{bmatrix}
   \begin{Bmatrix} \ddot u_b \\ \ddot u_i \end{Bmatrix}
   =
   \begin{Bmatrix} f_b \\ f_i \end{Bmatrix}.

Static condensation
-------------------

:func:`substructuring.static_condensation.static_condensation` keeps the
interface block :math:`u_b` and removes the interior. Each column of the basis
is the static response of the interior to a unit displacement of one interface
unknown,

.. math::

   K_{ii} \Psi = -K_{ib}, \qquad
   R = \begin{bmatrix} I \\ \Psi \end{bmatrix}.

The reduced stiffness is the Schur complement :math:`R^\mathsf{T} K R`, and
the mass is reduced by the same projection. A displacement of the component is
recovered from the interface alone, :math:`u = R u_b`. This is Guyan's
reduction [Guyan1965]_. Craig–Bampton starts from this basis and appends
fixed-interface modes. With ``n_modes = 0`` the two functions return the same
model.

Craig–Bampton reduction
-----------------------

Craig–Bampton uses two families of shapes [Craig1968]_.

A **constraint mode** is the static response to a unit displacement of one
interface unknown, with the other interface unknowns held at zero:

.. math::

   K_{ii} \Psi = -K_{ib}, \qquad \Psi_b = I.

With no vibration modes retained, :math:`R` is the static-condensation basis
above.

A **fixed-interface mode** is an eigenvector of the interior with the whole
interface of that component fixed,

.. math::

   K_{ii} \Phi = M_{ii} \Phi \Lambda, \qquad \Phi_b = 0.

Only the lowest ``n_modes`` eigenvectors are kept. The reduction basis and its
coordinates are

.. math::

   R =
   \begin{bmatrix}
   I & 0 \\
   \Psi & \Phi
   \end{bmatrix},
   \qquad
   q =
   \begin{Bmatrix} u_b \\ \eta \end{Bmatrix}.

"Fixed interface" means that the cuts belonging to this component are fixed.
It does not add supports that the component does not have.

- The root piece is already clamped at the wall, and its only cut is fixed, so
  its modes are those of a segment held at both ends.
- An interior piece has two cuts, and both are fixed.
- The tip piece has only the upstream cut fixed. Its free end remains free, so
  its modes are those of a short cantilever.

Keeping every interior eigenvector makes :math:`R` square and invertible, and
the coupled Craig–Bampton model reproduces the unreduced beam. The cantilever
script checks this on coarse meshes with two and three pieces. Setting
``n_modes = 0`` leaves the static constraint modes alone.

Rubin reduction
---------------

Rubin reduction starts from the free-interface vibration problem of the same
component [Rubin1975]_. The interface is not held: the eigenproblem is the full
free pair :math:`K \Phi = M \Phi \Lambda`, with mass-normalised eigenvectors,
:math:`\Phi^\mathsf{T} M \Phi = I`. The implementation is a primal
displacement form. Interface forces are not extra unknowns, which distinguishes
it from MacNeal's dual assembly.

A floating piece in the plane has three rigid-body modes: two translations and
one rotation. Those are the eigenvalues that sit at zero, separated from the
first elastic eigenvalue by the gap checked in the code. The root piece is
clamped, so it is called with ``n_rigid = 0``. Every other piece is called
with ``n_rigid = 3``. The columns are split into rigid modes :math:`\Phi_r`,
kept elastic modes :math:`\Phi_k` with eigenvalues :math:`\Lambda_k`, and the
elastic modes that are discarded.

The flexibility that remains after this split is built in two steps. First the
inertia-relief flexibility :math:`G` removes the rigid motion. If the piece has
no rigid modes, :math:`G = K^{-1}`. Otherwise the mass-normalised rigid modes
are used to make the stiffness positive definite and to keep only the part of
a load that does no rigid work,

.. math::

   K_\star = K + (M \Phi_r)(M \Phi_r)^\mathsf{T},
   \qquad
   K_\star G = I - M \Phi_r \Phi_r^\mathsf{T}.

The kept elastic modes are then subtracted, leaving the residual flexibility of
everything that was discarded:

.. math::

   G_{\mathrm{res}} = G - \Phi_k \Lambda_k^{-1} \Phi_k^\mathsf{T}.

An **attachment mode** spreads that residual flexibility onto the interface and
scales it so that the interface trace is the identity,

.. math::

   \Psi = G_{\mathrm{res},\, :b}\, G_{\mathrm{res},\, bb}^{-1},
   \qquad \Psi_b = I.

At least one elastic mode has to stay outside the basis. If every elastic mode
were kept, :math:`G_{\mathrm{res}}` would vanish and these attachment modes
would not exist. A Rubin model with all of its modes is therefore not a
completeness test, unlike Craig–Bampton.

The free-interface shapes still move the cut. That motion is taken out of the
modal columns and left to the attachment modes,

.. math::

   \hat\Phi = \bigl[\Phi_r \;\; \Phi_k\bigr]
   - \Psi \bigl[\Phi_r \;\; \Phi_k\bigr]_b.

The basis stored for the assembly is :math:`R = [\Psi \;\; \hat\Phi]`. The
shapes kept for plotting are the original columns :math:`\Phi_r` and
:math:`\Phi_k`, before this deflation, because those are the physical
free-interface motions.

Assembly of a chain
-------------------

:func:`substructuring.coupling.couple` accepts the reduced pieces in order from
the wall to the tip. The first and the last piece each have one interface
block. Each interior piece has its left cut followed by its right cut, and the
two sides of a cut must have the same length.

The coupled unknown lists every cut, from the wall toward the tip, and then
the local coordinates of piece 0, piece 1, and so on:

.. math::

   q =
   \bigl(
   u^{(0)}, \ldots, u^{(N-2)},\;
   \eta^{(0)}, \ldots, \eta^{(N-1)}
   \bigr).

Each reduced matrix is added into the rows and columns of the cuts it touches
and of its own modal coordinates. Shared cut displacements enforce the
continuity of the chain. The coupled size is the number of cut degrees of
freedom plus the number of retained local coordinates.

With the same number of elastic modes on every piece, Craig–Bampton keeps that
many modal coordinates per piece. Rubin keeps the same elastic coordinates and,
on every piece that is not the root, three additional rigid-body coordinates.
The interface block is present either way, so refining the cut from two pieces
into four or five adds a full set of cut unknowns for each new interface. That
is the growth of the reduced model with the number of substructures.

References
----------

.. [Craig1968] R. R. Craig Jr. and M. C. C. Bampton, "Coupling of substructures
   for dynamic analyses", *AIAA Journal* 6(7), 1313–1319, 1968.
   `doi:10.2514/3.4741 <https://doi.org/10.2514/3.4741>`_.

.. [Rubin1975] S. Rubin, "Improved component-mode representation for structural
   dynamic analysis", *AIAA Journal* 13(8), 995–1006, 1975.
   `doi:10.2514/3.60497 <https://doi.org/10.2514/3.60497>`_.

.. [Guyan1965] R. J. Guyan, "Reduction of stiffness and mass matrices",
   *AIAA Journal* 3(2), 380, 1965.
   `doi:10.2514/3.2874 <https://doi.org/10.2514/3.2874>`_.
