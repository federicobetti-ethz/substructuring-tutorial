"""Rubin reduction with free-interface modes and residual attachment modes."""

import numpy as np
from numpy.typing import NDArray
from scipy.linalg import eigh, solve

from substructuring.coupling import ReducedModel


def rubin(
    stiffness: NDArray[np.float64],
    mass: NDArray[np.float64],
    boundary: NDArray[np.int64],
    n_modes: int,
    n_rigid: int = 0,
) -> ReducedModel:
    """Reduce one component with free-interface modes and residual attachment modes.

    Free-interface modes are the vibration modes of the component with its
    interface left free. A floating component also contributes ``n_rigid``
    rigid-body modes; a clamped component uses ``n_rigid=0``. The residual
    flexibility is the inertia-relief flexibility of every discarded elastic
    mode. Its action on the interface, scaled to a unit interface displacement,
    supplies the attachment modes that carry the interface motion. At least one
    elastic mode has to stay out of the basis, otherwise that residual vanishes.
    """
    size = stiffness.shape[0]
    if n_modes < 0 or n_rigid < 0:
        raise ValueError("Mode counts must be non-negative")
    if n_rigid + n_modes >= size:
        raise ValueError("Rubin needs at least one discarded elastic mode")
    if boundary.size == 0:
        raise ValueError("Rubin needs a non-empty interface")

    eigenvalues, vectors = eigh(stiffness, mass)
    _check_rigid_spectrum(eigenvalues, n_rigid)
    rigid = vectors[:, :n_rigid]
    kept = vectors[:, n_rigid : n_rigid + n_modes]
    kept_stiffness = eigenvalues[n_rigid : n_rigid + n_modes]

    flexibility = _inertia_relief_flexibility(stiffness, mass, rigid)
    if n_modes:
        flexibility = flexibility - (kept / kept_stiffness) @ kept.T
    flexibility = 0.5 * (flexibility + flexibility.T)

    attachment = solve(
        flexibility[np.ix_(boundary, boundary)],
        flexibility[:, boundary].T,
        assume_a="sym",
    ).T
    free_modes = np.hstack([rigid, kept])
    free_modes = free_modes - attachment @ free_modes[boundary]
    basis = np.hstack([attachment, free_modes])
    _assert_interface_partition(basis, boundary)
    return ReducedModel(
        stiffness=basis.T @ stiffness @ basis,
        mass=basis.T @ mass @ basis,
        basis=basis,
        n_boundary=boundary.size,
    )


def _inertia_relief_flexibility(
    stiffness: NDArray[np.float64],
    mass: NDArray[np.float64],
    rigid: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Flexibility that does not resist rigid motion.

    With mass-normalized rigid modes the augmented stiffness
    ``K + M Φr Φr.T M`` is positive definite, and removing ``M Φr Φr.T`` from
    the load leaves the self-balanced part of a force.
    """
    if rigid.shape[1] == 0:
        return solve(stiffness, np.eye(stiffness.shape[0]), assume_a="pos")
    inertia = mass @ rigid
    augmented = stiffness + inertia @ inertia.T
    load = np.eye(stiffness.shape[0]) - inertia @ rigid.T
    flexibility = solve(augmented, load, assume_a="sym")
    return 0.5 * (flexibility + flexibility.T)


def _check_rigid_spectrum(eigenvalues: NDArray[np.float64], n_rigid: int) -> None:
    if n_rigid == 0:
        if eigenvalues[0] <= 0.0:
            raise RuntimeError("The component is singular; pass its rigid-body count")
        return
    rigid = np.max(np.abs(eigenvalues[:n_rigid]))
    elastic = eigenvalues[n_rigid]
    if elastic <= 0.0 or rigid > 1e-6 * elastic:
        raise RuntimeError(
            f"Expected {n_rigid} rigid-body eigenvalues near zero, "
            f"found a gap of {rigid:.3e} before the first elastic value {elastic:.3e}"
        )


def _assert_interface_partition(basis: NDArray[np.float64], boundary: NDArray[np.int64]) -> None:
    n_boundary = boundary.size
    identity = basis[boundary, :n_boundary]
    modal = basis[boundary, n_boundary:]
    if not np.allclose(identity, np.eye(n_boundary), atol=1e-7):
        raise RuntimeError("Attachment modes do not form an identity on the interface")
    if modal.size and not np.allclose(modal, 0.0, atol=1e-7):
        raise RuntimeError("Free-interface coordinates still move the interface")
