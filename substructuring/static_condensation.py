"""Guyan reduction by static condensation of the interior."""

import numpy as np
from numpy.typing import NDArray
from scipy.linalg import solve

from substructuring.shapes import ComponentReduction, ModeShape


def static_condensation(
    stiffness: NDArray[np.float64],
    mass: NDArray[np.float64],
    boundary: NDArray[np.int64],
) -> ComponentReduction:
    r"""Keep the interface unknowns and condense the interior.

    The basis columns are the constraint modes. The interior block solves

    .. math::

       K_{ii}\Psi = -K_{ib}, \qquad
       R = \begin{bmatrix} I \\ \Psi \end{bmatrix}.

    A displacement of the component is recovered by :math:`u = R u_b`.
    This is the Guyan reduction: the reduced stiffness is the Schur
    complement :math:`R^\mathsf{T} K R` of the interior, and the mass is
    reduced with the same basis.

    Parameters
    ----------
    stiffness
        Free stiffness of the component.
    mass
        Consistent mass of the component, in the same degrees of freedom.
    boundary
        Indices of the interface unknowns that stay in the reduced model.

    Returns
    -------
    ComponentReduction
        Interface coordinates only.
    """
    interior = np.setdiff1d(
        np.arange(stiffness.shape[0]), boundary, assume_unique=False
    )

    constraint = np.zeros((stiffness.shape[0], boundary.size))
    constraint[boundary] = np.eye(boundary.size)
    constraint[interior] = solve(
        stiffness[np.ix_(interior, interior)],
        -stiffness[np.ix_(interior, boundary)],
        assume_a="pos",
    )

    return ComponentReduction(
        stiffness=constraint.T @ stiffness @ constraint,
        mass=constraint.T @ mass @ constraint,
        basis=constraint,
        n_boundary=boundary.size,
        shapes=(ModeShape("constraint", constraint),),
    )
