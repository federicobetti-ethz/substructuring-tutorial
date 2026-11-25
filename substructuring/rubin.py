"""Rubin reduction with free-interface modes and residual attachment modes."""

import numpy as np
from numpy.typing import NDArray
from scipy.linalg import eigh, solve

from substructuring.coupling import ModeShape, ReducedModel


def rubin(
    stiffness: NDArray[np.float64],
    mass: NDArray[np.float64],
    boundary: NDArray[np.int64],
    n_modes: int,
    n_rigid: int = 0,
) -> ReducedModel:
    r"""Reduce one component with free-interface modes and residual attachment modes.

    Free-interface modes are the vibration modes of the component with its
    interface left free,

    .. math::

       K\Phi = M\Phi\Lambda, \qquad \Phi^\mathsf{T} M \Phi = I.

    A floating component also contributes ``n_rigid`` rigid-body modes; a
    clamped component uses ``n_rigid = 0``. The residual flexibility is the
    inertia-relief flexibility of every discarded elastic mode,

    .. math::

       G_{\mathrm{res}} = G - \Phi_k \Lambda_k^{-1} \Phi_k^\mathsf{T}.

    Its action on the interface, scaled to a unit interface displacement,
    supplies the attachment modes

    .. math::

       \Psi = G_{\mathrm{res},\,:b}\, G_{\mathrm{res},\,bb}^{-1},
       \qquad \Psi_b = I.

    At least one elastic mode has to stay out of the basis, otherwise that
    residual vanishes.

    Parameters
    ----------
    stiffness
        Free stiffness of the component.
    mass
        Consistent mass of the component, in the same degrees of freedom.
    boundary
        Indices of the interface unknowns.
    n_modes
        Number of elastic free-interface modes to keep, lowest frequencies
        first.
    n_rigid
        Number of rigid-body modes to keep before the elastic ones. Use 0
        for a clamped component and 3 for a floating piece in the plane.

    Returns
    -------
    ReducedModel
        Coordinates :math:`(u_b, \eta)`. Rigid-body coordinates are part of
        :math:`\eta` when ``n_rigid`` is not zero. The basis maps the reduced
        coordinates to the free finite-element displacement by :math:`u = R q`.
    """
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

    shapes = [ModeShape("attachment", attachment)]
    
    if n_rigid:
        shapes.append(ModeShape("rigid body", rigid, np.zeros(n_rigid)))
    
    if n_modes:
        shapes.append(ModeShape("free interface", kept, _hertz(kept_stiffness)))
    
    free_modes = np.hstack([rigid, kept])
    free_modes = free_modes - attachment @ free_modes[boundary]
    basis = np.hstack([attachment, free_modes])
    
    _assert_interface_partition(basis, boundary)
    
    return ReducedModel(
        stiffness=basis.T @ stiffness @ basis,
        mass=basis.T @ mass @ basis,
        basis=basis,
        n_boundary=boundary.size,
        shapes=tuple(shapes),
    )


def _inertia_relief_flexibility(
    stiffness: NDArray[np.float64],
    mass: NDArray[np.float64],
    rigid: NDArray[np.float64],
) -> NDArray[np.float64]:
    r"""Flexibility that does not resist rigid motion.

    With mass-normalised rigid modes the augmented stiffness

    .. math::

       K_\star = K + (M\Phi_r)(M\Phi_r)^\mathsf{T}

    is positive definite, and the load :math:`I - M\Phi_r\Phi_r^\mathsf{T}`
    keeps only the part of a force that does no rigid work,

    .. math::

       K_\star G = I - M\Phi_r\Phi_r^\mathsf{T}.

    Parameters
    ----------
    stiffness
        Free stiffness of the component.
    mass
        Consistent mass of the component, in the same degrees of freedom.
    rigid
        Mass-normalised rigid-body modes, one column each. An empty array
        selects the inverse of ``stiffness``.

    Returns
    -------
    ndarray
        Inertia-relief flexibility :math:`G`.
    """
    if rigid.shape[1] == 0:
        return solve(stiffness, np.eye(stiffness.shape[0]), assume_a="pos")
    inertia = mass @ rigid
    augmented = stiffness + inertia @ inertia.T
    load = np.eye(stiffness.shape[0]) - inertia @ rigid.T
    flexibility = solve(augmented, load, assume_a="sym")
    return 0.5 * (flexibility + flexibility.T)


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


def _check_rigid_spectrum(eigenvalues: NDArray[np.float64], n_rigid: int) -> None:
    r"""Require the requested rigid-body eigenvalues to sit at zero.

    Parameters
    ----------
    eigenvalues
        Eigenvalues of :math:`K\Phi = M\Phi\lambda`, in ascending order.
    n_rigid
        Number of rigid-body modes expected at the start of ``eigenvalues``.
        Zero requires the component to be positive definite.

    Raises
    ------
    RuntimeError
        If ``n_rigid`` is zero and the smallest eigenvalue is not positive,
        or if the first ``n_rigid`` eigenvalues are not separated from the
        first elastic eigenvalue by the gap checked in the code.
    """
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
    """Require attachment modes to be the identity and free-interface coordinates to vanish on the cut.

    Parameters
    ----------
    basis
        Rubin basis. The leading columns are the attachment modes and the
        remaining columns are the deflated rigid-body and free-interface modes.
    boundary
        Indices of the interface unknowns.

    Raises
    ------
    RuntimeError
        If the attachment-mode block on ``boundary`` is not the identity,
        or if a free-interface coordinate still moves an interface unknown.
    """
    n_boundary = boundary.size
    identity = basis[boundary, :n_boundary]
    modal = basis[boundary, n_boundary:]
    if not np.allclose(identity, np.eye(n_boundary), atol=1e-7):
        raise RuntimeError("Attachment modes do not form an identity on the interface")
    if modal.size and not np.allclose(modal, 0.0, atol=1e-7):
        raise RuntimeError("Free-interface coordinates still move the interface")
