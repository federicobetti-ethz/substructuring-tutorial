"""Guyan reduction by static condensation of the interior."""

import numpy as np
from numpy.typing import NDArray
from scipy.linalg import solve

from substructuring.coupling import ModeShape, ReducedModel


def static_condensation(
    stiffness: NDArray[np.float64],
    mass: NDArray[np.float64],
    boundary: NDArray[np.int64],
) -> ReducedModel:
    """Keep the interface unknowns and condense the interior.

    The basis columns are the constraint modes. The interior block solves
    ``K_ii Ψ = -K_ib``, and the basis is ``[I; Ψ]``. A displacement of the
    component is recovered by ``u = basis @ u_b``. This is the Guyan reduction:
    the reduced stiffness is the Schur complement of the interior, and the mass
    is reduced with the same basis. Craig–Bampton adds fixed-interface modes
    to this basis.
    """
    if boundary.size == 0:
        raise ValueError("Static condensation needs an interface to retain")
    interior = np.setdiff1d(np.arange(stiffness.shape[0]), boundary, assume_unique=False)
    if interior.size == 0:
        raise ValueError("The component has no interior degrees of freedom")

    constraint = np.zeros((stiffness.shape[0], boundary.size))
    constraint[boundary] = np.eye(boundary.size)
    constraint[interior] = solve(
        stiffness[np.ix_(interior, interior)],
        -stiffness[np.ix_(interior, boundary)],
        assume_a="pos",
    )
    return ReducedModel(
        stiffness=constraint.T @ stiffness @ constraint,
        mass=constraint.T @ mass @ constraint,
        basis=constraint,
        n_boundary=boundary.size,
        shapes=(ModeShape("constraint", constraint),),
    )
