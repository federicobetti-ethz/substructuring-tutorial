Theory
======

A substructure is essentially a contiguous subset of the nodes of the mesh,
with its own stiffness matrix :math:`K` and mass matrix :math:`M`.
Component-mode synthesis replaces the unknowns inside that
piece by a few generalised coordinates, while the displacements on each cut
stay as the coordinates that tie the pieces together. The forms below are the
ones implemented in :func:`substructuring.craig_bampton.craig_bampton` and
:func:`substructuring.rubin.rubin`. Both return a basis :math:`R` such that
the physical displacement is :math:`u = R q`, and both reduce the matrices by
the same projection,

.. math::

   \widetilde{K} = R^\mathsf{T} K R, \qquad \widetilde{M} = R^\mathsf{T} M R.

The finite-element component
----------------------------

The example employed in this codebase is a plane-stress rectangular beam. Given the
Young's modulus :math:`E` and Poisson's ratio :math:`\nu`, the Lamé parameters are given by

.. math::

   \mu = \frac{E}{2(1+\nu)}, \qquad
   \lambda = \frac{E \nu}{1-\nu^2}.

The example is meshed with quadratic quadrilaterals, and the weak
form, scaled by the thickness :math:`t`, is

.. math::

   a(u, v) = \int_\Omega t \bigl( 2\mu\, \varepsilon(u)\!:\!\varepsilon(v)
   + \lambda\, \operatorname{tr}\varepsilon(u)\, \operatorname{tr}\varepsilon(v) \bigr) \,\mathrm{d}A,

and the mass form is :math:`\int_\Omega t \rho\, u \cdot v \,\mathrm{d}A`.
Clamped degrees of freedom are removed before the reduction, so :math:`K` is
the free stiffness of that piece.

The free unknowns are split into the interface block :math:`u_b` and the
interior block :math:`u_i`. The partitioned equation, still without damping, is

.. math::
   
   \begin{bmatrix}
   M_{bb} & M_{bi} \\
   M_{ib} & M_{ii}
   \end{bmatrix}
   \begin{Bmatrix} \ddot u_b \\ \ddot u_i \end{Bmatrix}
   +
   \begin{bmatrix}
   K_{bb} & K_{bi} \\
   K_{ib} & K_{ii}
   \end{bmatrix}
   \begin{Bmatrix} u_b \\ u_i \end{Bmatrix}
   =
   \begin{Bmatrix} f_b \\ f_i \end{Bmatrix}.

Primal assembly
---------------

The substructure matrices stay separate. Stacking them on the diagonal keeps
a redundant copy of every cut,

.. math::

   M = \operatorname{diag}\bigl(M^{(1)}, \ldots, M^{(N)}\bigr), \qquad
   K = \operatorname{diag}\bigl(K^{(1)}, \ldots, K^{(N)}\bigr), \qquad
   u = \begin{Bmatrix} u^{(1)} \\ \vdots \\ u^{(N)} \end{Bmatrix}.

Each cut equates the two local copies of one displacement. Those constraints
are the rows of a signed Boolean matrix :math:`B`, and a connected
displacement satisfies :math:`Bu = 0`. The localization :math:`L` is the
Boolean basis of that kernel, :math:`u = Lq`, so :math:`BL = 0`. A connecting
force :math:`g = B^\mathsf{T}\lambda` then does no work on the primal
coordinates,

.. math::

   L^\mathsf{T} g = 0,

and the coupled matrices are the products

.. math::

   \widetilde{K} = L^\mathsf{T} K L, \qquad \widetilde{M} = L^\mathsf{T} M L.

:class:`~substructuring.substructure.Substructure` stores one diagonal block.
Its interior and its boundary are numbered twice: locally, as rows of
:math:`K^{(s)}`, and globally, as columns of :math:`L`.
:meth:`~substructuring.substructure.Substructure.reduce` replaces that block
by a Guyan, Craig–Bampton, or Rubin reduction. The boundary keeps its global
numbers, and each new modal coordinate receives a global number of its own.
:class:`~substructuring.assembly.GlobalModel` builds :math:`L` and :math:`B`
from those numbers and forms the product above.

Static condensation
-------------------

:func:`substructuring.static_condensation.static_condensation` keeps the
interface block :math:`u_b` and removes the interior. Each column of the basis
is the static response of the interior to a unit displacement of one interface
unknown,

.. math::

   K_{ii}^{(s)} \Psi^{(s)} = -K_{ib}^{(s)}, \qquad
   R^{(s)} = \begin{bmatrix} I \\ \Psi^{(s)} \end{bmatrix}.

The reduced stiffness is the Schur complement :math:`R^{(s)\mathsf{T}} K^{(s)} R^{(s)}`, and
the mass is reduced by the same projection. A displacement of the substructure is
recovered from the interface alone, :math:`u = R^{(s)} u_b^{(s)}`. This is Guyan's
reduction [Guyan1965]_.

Craig–Bampton reduction
-----------------------

Craig–Bampton uses two families of shapes [Craig1968]_. In a nutshell,
it starts from the static condensation basis and appends fixed-interface modes.

In the Craig–Bampton setting, the static condensation basis is said to consist 
of **constraint modes**. These are the static responses to a unit displacement of one
interface unknown, with the other interface unknowns held at zero:

.. math::

   K_{ii}^{(s)} \Psi^{(s)} = -K_{ib}^{(s)}.

A **fixed-interface mode** is an eigenvector of the interior with the whole
interface of that substructure fixed,

.. math::

   K_{ii}^{(s)} \Phi^{(s)} = M_{ii}^{(s)} \Phi^{(s)} \Lambda^{(s)}.

Only the lowest ``n_modes`` eigenvectors are kept, depending on the frequency range of interest.
The reduction basis and its coordinates are given by

.. math::

   R^{(s)} =
   \begin{bmatrix}
   I & 0 \\
   \Psi^{(s)} & \Phi^{(s)}
   \end{bmatrix},
   \qquad
   q =
   \begin{Bmatrix} u_b^{(s)} \\ \eta^{(s)} \end{Bmatrix}.

Keeping every interior eigenvector makes :math:`R^{(s)}` square and invertible, and
the coupled Craig–Bampton model reproduces the full model of the substructure.

Rubin reduction
---------------

Rubin reduction starts from the free-interface vibration problem of the same
substructure [Rubin1975]_. The interface is not held: the eigenproblem is the full

.. math::
   
   K^{(s)} \Phi^{(s)} = M^{(s)} \Phi^{(s)} \Lambda^{(s)},
   \qquad
   \Phi^{(s)\mathsf{T}} M^{(s)} \Phi^{(s)} = I.

Rigid-body modes are the eigenvalues that sit at zero, separated from the
first elastic eigenvalue by the gap checked in the code. They are part of the basis,
but they are removed when computing the remaining flexibility matrix.

The latter that remains after this split is built in two steps. First the
inertia-relief flexibility :math:`G` removes the rigid motion. If the piece has
no rigid modes, :math:`G = K^{-1}`. Otherwise the mass-normalised rigid modes
are used to make the stiffness positive definite and to keep only the part of
a load that does no rigid work,

.. math::

   K_\star = K^{(s)} + (M^{(s)} \Phi_r^{(s)})(M^{(s)} \Phi_r^{(s)})^\mathsf{T},
   \qquad
   K_\star^{(s)} G^{(s)} = I - M^{(s)} \Phi_r^{(s)} \Phi_r^{(s)\mathsf{T}}.

The kept elastic modes are then subtracted, leaving the residual flexibility of
everything that was discarded:

.. math::

   G_{\mathrm{res}}^{(s)} = G^{(s)} - \Phi_k^{(s)} \Lambda_k^{(s)}^{-1} \Phi_k^{(s)\mathsf{T}}.

An **attachment mode** spreads that residual flexibility onto the interface and
scales it so that the interface trace is the identity,

.. math::

   \Psi^{(s)} = G_{\mathrm{res},\, :b}^{(s)}\, G_{\mathrm{res},\, bb}^{(s)}^{-1}.

At least one elastic mode has to stay outside the basis. If every elastic mode
were kept, :math:`G_{\mathrm{res}}` would vanish and these attachment modes
would not exist. A Rubin model with all of its modes is therefore not a
completeness test, unlike Craig–Bampton.

The free-interface shapes still move the cut. That motion is taken out of the
modal columns and left to the attachment modes,

.. math::

   \hat\Phi^{(s)} = \bigl[\Phi_r^{(s)} \;\; \Phi_k^{(s)}\bigr]
   - \Psi^{(s)} \bigl[\Phi_r^{(s)} \;\; \Phi_k^{(s)}\bigr]_b.

The basis stored for the assembly is :math:`R^{(s)} = [\Psi^{(s)} \;\; \hat\Phi^{(s)}]`. The
shapes kept for plotting are the original columns :math:`\Phi_r^{(s)}` and
:math:`\Phi_k`, before this deflation, because those are the physical
free-interface motions.

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
