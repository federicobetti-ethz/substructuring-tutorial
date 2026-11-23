"""Craig–Bampton reduction with fixed-interface modes."""

import numpy as np
from numpy.typing import NDArray
from scipy.linalg import eigh, solve

from substructuring.coupling import ModeShape, ReducedModel


def craig_bampton(
    stiffness: NDArray[np.float64],
    mass: NDArray[np.float64],
    boundary: NDArray[np.int64],
    n_modes: int,
) -> ReducedModel:
    """Reduce one component with constraint modes and fixed-interface modes.

    The interface is held fixed while the interior eigenproblem is solved, and
    each constraint mode is the static response to a unit displacement of one
    interface unknown. Keeping every interior mode reproduces the original
    component. ``n_modes`` may be zero, which leaves the Guyan reduction.
    """
    if n_modes < 0:
        raise ValueError("n_modes must be non-negative")
    interior = np.setdiff1d(np.arange(stiffness.shape[0]), boundary, assume_unique=False)
    if interior.size == 0:
        raise ValueError("The component has no interior degrees of freedom")
    if n_modes > interior.size:
        raise ValueError(
            f"Requested {n_modes} fixed-interface modes, but only {interior.size} exist"
        )

    interior_stiffness = stiffness[np.ix_(interior, interior)]
    constraint = np.zeros((stiffness.shape[0], boundary.size))
    constraint[boundary] = np.eye(boundary.size)
    constraint[interior] = solve(
        interior_stiffness,
        -stiffness[np.ix_(interior, boundary)],
        assume_a="pos",
    )

    shapes = [ModeShape("constraint", constraint)]
    columns = [constraint]
    if n_modes:
        modes = np.zeros((stiffness.shape[0], n_modes))
        values, vectors = eigh(
            interior_stiffness,
            mass[np.ix_(interior, interior)],
            subset_by_index=[0, n_modes - 1],
        )
        if np.any(values <= 0.0):
            raise RuntimeError("The fixed interface is not restraining the component")
        modes[interior] = vectors
        columns.append(modes)
        shapes.append(ModeShape("fixed interface", modes, _hertz(values)))

    basis = np.hstack(columns)
    _assert_interface_partition(basis, boundary)
    return ReducedModel(
        stiffness=basis.T @ stiffness @ basis,
        mass=basis.T @ mass @ basis,
        basis=basis,
        n_boundary=boundary.size,
        shapes=tuple(shapes),
    )


def _hertz(stiffness_eigenvalues: NDArray[np.float64]) -> NDArray[np.float64]:
    return np.sqrt(np.maximum(stiffness_eigenvalues, 0.0)) / (2.0 * np.pi)


def _assert_interface_partition(basis: NDArray[np.float64], boundary: NDArray[np.int64]) -> None:
    n_boundary = boundary.size
    identity = basis[boundary, :n_boundary]
    modal = basis[boundary, n_boundary:]
    if not np.allclose(identity, np.eye(n_boundary), atol=1e-8):
        raise RuntimeError("Constraint modes do not form an identity on the interface")
    if modal.size and not np.allclose(modal, 0.0, atol=1e-8):
        raise RuntimeError("Fixed-interface modes move the interface")
