"""Craig–Bampton reduction with fixed-interface modes."""

import numpy as np
from numpy.typing import NDArray
from scipy.linalg import eigh

from substructuring.shapes import ComponentReduction, ModeShape
from substructuring.static_condensation import static_condensation


def craig_bampton(
    stiffness: NDArray[np.float64],
    mass: NDArray[np.float64],
    boundary: NDArray[np.int64],
    n_modes: int,
) -> ComponentReduction:
    r"""Reduce one component with constraint modes and fixed-interface modes.

    The constraint modes are the static-condensation basis,

    .. math::

       K_{ii}\Psi = -K_{ib}, \qquad
       R = \begin{bmatrix} I \\ \Psi \end{bmatrix}.

    The interface is held fixed while the interior eigenproblem is solved,

    .. math::

       K_{ii}\Phi = M_{ii}\Phi\Lambda, \qquad \Phi_b = 0.

    Only the lowest ``n_modes`` eigenvectors are appended to :math:`R`.
    Keeping every interior mode reproduces the original component.
    ``n_modes`` may be zero, which returns
    :func:`~substructuring.static_condensation.static_condensation`.

    Parameters
    ----------
    stiffness
        Free stiffness of the component.
    mass
        Consistent mass of the component, in the same degrees of freedom.
    boundary
        Indices of the interface unknowns.
    n_modes
        Number of fixed-interface modes to keep, lowest frequencies first.
        Zero keeps the constraint modes only.

    Returns
    -------
    ComponentReduction
        Coordinates :math:`(u_b, \eta)`. The basis maps them to the free
        finite-element displacement by :math:`u = R q`.
    """
    condensed = static_condensation(stiffness, mass, boundary)
    if n_modes == 0:
        return condensed
    interior = np.setdiff1d(
        np.arange(stiffness.shape[0]), boundary, assume_unique=False
    )

    interior_stiffness = stiffness[np.ix_(interior, interior)]
    constraint = condensed.basis
    modes = np.zeros((stiffness.shape[0], n_modes))

    values, vectors = eigh(
        interior_stiffness,
        mass[np.ix_(interior, interior)],
        subset_by_index=[0, n_modes - 1],
    )

    modes[interior] = vectors
    basis = np.hstack((constraint, modes))
    _assert_interface_partition(basis, boundary)

    return ComponentReduction(
        stiffness=basis.T @ stiffness @ basis,
        mass=basis.T @ mass @ basis,
        basis=basis,
        n_boundary=boundary.size,
        shapes=(
            ModeShape("constraint", constraint),
            ModeShape("fixed interface", modes, _hertz(values)),
        ),
    )


def _hertz(stiffness_eigenvalues: NDArray[np.float64]) -> NDArray[np.float64]:
    r"""Convert stiffness eigenvalues to frequencies in hertz.

    Parameters
    ----------
    stiffness_eigenvalues
        Eigenvalues :math:`\lambda` of :math:`K\Phi = M\Phi\lambda`. Negative
        values are clipped to zero before the square root.

    Returns
    -------
    ndarray
        Frequencies :math:`\sqrt{\lambda}/(2\pi)`.
    """
    return np.sqrt(np.maximum(stiffness_eigenvalues, 0.0)) / (2.0 * np.pi)


def _assert_interface_partition(
    basis: NDArray[np.float64], boundary: NDArray[np.int64]
) -> None:
    """Require constraint modes to be the identity and fixed-interface modes to vanish on the cut.

    Parameters
    ----------
    basis
        Craig–Bampton basis. The leading columns are the constraint modes
        and the remaining columns are the fixed-interface modes.
    boundary
        Indices of the interface unknowns.

    Raises
    ------
    RuntimeError
        If the constraint-mode block on ``boundary`` is not the identity,
        or if a fixed-interface mode moves an interface unknown.
    """
    n_boundary = boundary.size
    identity = basis[boundary, :n_boundary]
    modal = basis[boundary, n_boundary:]
    if not np.allclose(identity, np.eye(n_boundary), atol=1e-8):
        raise RuntimeError("Constraint modes do not form an identity on the interface")
    if modal.size and not np.allclose(modal, 0.0, atol=1e-8):
        raise RuntimeError("Fixed-interface modes move the interface")
