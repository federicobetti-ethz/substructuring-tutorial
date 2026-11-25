"""Plane-stress assembly on a quadratic quadrilateral mesh."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.linalg import eigh
from skfem import Basis, BilinearForm, ElementQuad2, ElementVector, MeshLine, asm
from skfem.helpers import ddot, dot, sym_grad, trace
from skfem.mesh import Mesh

from substructuring.coupling import ModeShape


@dataclass(frozen=True)
class Material:
    r"""Isotropic plane-stress material.

    Displacements are in metres when the mesh coordinates are in metres.

    Attributes
    ----------
    young
        Young's modulus :math:`E`.
    poisson
        Poisson's ratio :math:`\nu`.
    density
        Mass density :math:`\rho`.
    thickness
        Out-of-plane thickness :math:`t`. Stiffness and mass are scaled by it.
    """

    young: float
    poisson: float
    density: float
    thickness: float


@dataclass(frozen=True)
class Model:
    r"""Stiffness and mass with prescribed displacements already removed.

    ``boundary`` indexes the interface degrees of freedom that component-mode
    synthesis keeps. The interface block lists every horizontal unknown in
    order of increasing :math:`y`, then every vertical unknown in the same
    order, so two components that share a cut can be coupled entry by entry.

    Attributes
    ----------
    stiffness
        Free stiffness :math:`K`.
    mass
        Free mass :math:`M`, in the same degrees of freedom as ``stiffness``.
    boundary
        Indices of the interface unknowns. An interior piece stores the left
        cut first and the right cut second.
    coordinates
        Coordinates of the free degrees of freedom, one row per unknown and
        columns :math:`(x, y)`.
    component
        ``0`` for a horizontal unknown and ``1`` for a vertical unknown.
    vertex_dofs
        Free degree-of-freedom index at each mesh vertex, or ``-1`` when that
        unknown was removed by the clamp. Shape ``(n_vertices, 2)``.
    """

    stiffness: NDArray[np.float64]
    mass: NDArray[np.float64]
    boundary: NDArray[np.int64]
    coordinates: NDArray[np.float64]
    component: NDArray[np.int64]
    vertex_dofs: NDArray[np.int64]


def rectangle_mesh(x0: float, x1: float, height: float, nx: int, ny: int) -> Mesh:
    r"""Tensor-product quadrilateral mesh of :math:`[x_0, x_1] \times [0, h]`.

    Parameters
    ----------
    x0
        Left edge :math:`x_0`.
    x1
        Right edge :math:`x_1`.
    height
        Top edge :math:`h`. The bottom edge is :math:`y = 0`.
    nx
        Number of elements along :math:`x`.
    ny
        Number of elements along :math:`y`.

    Returns
    -------
    Mesh
        Quadrilateral mesh with boundaries ``left``, ``right``, ``bottom``,
        and ``top``.
    """
    if nx < 1 or ny < 1:
        raise ValueError("nx and ny must be positive")
    horizontal = MeshLine(np.linspace(x0, x1, nx + 1))
    vertical = MeshLine(np.linspace(0.0, height, ny + 1))
    return (horizontal * vertical).with_defaults()


def assemble_plane_stress(
    mesh: Mesh, material: Material
) -> tuple[NDArray[np.float64], NDArray[np.float64], Basis]:
    r"""Return dense stiffness, dense mass, and the quadratic vector basis.

    The constitutive law is plane stress. With shear modulus
    :math:`\mu = E/(2(1+\nu))` and plane-stress modulus
    :math:`\lambda = E\nu/(1-\nu^2)`,

    .. math::

       a(u, v) = \int_\Omega t \bigl(
       2\mu\, \varepsilon(u)\!:\!\varepsilon(v)
       + \lambda\, \operatorname{tr}\varepsilon(u)\,
       \operatorname{tr}\varepsilon(v) \bigr) \,\mathrm{d}A.

    The mass form is :math:`\int_\Omega t\rho\, u\cdot v \,\mathrm{d}A`.
    Both matrices are scaled by the thickness, so they are the stiffness and
    mass of a flat strip. Clamped degrees of freedom are still present.

    Parameters
    ----------
    mesh
        Quadrilateral mesh of the component.
    material
        Isotropic plane-stress constants.

    Returns
    -------
    stiffness : ndarray
        Dense stiffness, including degrees of freedom that a clamp would remove.
    mass : ndarray
        Dense mass, in the same ordering as ``stiffness``.
    basis : Basis
        Quadratic vector basis used to number the degrees of freedom.
    """
    basis = Basis(mesh, ElementVector(ElementQuad2()))
    shear = material.young / (2.0 * (1.0 + material.poisson))
    lame = material.young * material.poisson / (1.0 - material.poisson**2)
    weight = material.thickness

    @BilinearForm
    def stiffness(u, v, w):
        strain_u = sym_grad(u)
        strain_v = sym_grad(v)
        return weight * (
            2.0 * shear * ddot(strain_u, strain_v)
            + lame * trace(strain_u) * trace(strain_v)
        )

    @BilinearForm
    def mass(u, v, w):
        return weight * material.density * dot(u, v)

    stiffness_matrix = _symmetrize(asm(stiffness, basis).toarray())
    mass_matrix = _symmetrize(asm(mass, basis).toarray())
    return stiffness_matrix, mass_matrix, basis


def structural_model(
    mesh: Mesh,
    material: Material,
    interface: str | tuple[str, ...] | None = None,
    clamp: str | None = None,
) -> Model:
    """Assemble one component and remove the clamped degrees of freedom.

    ``interface`` and ``clamp`` are boundary names created by
    :func:`rectangle_mesh`: ``left``, ``right``, ``bottom``, or ``top``.
    A tuple lists the cuts in the order stored in the boundary block. A chain
    uses ``"right"`` on the first piece, ``("left", "right")`` on interior
    pieces, and ``"left"`` on the last piece. Omit ``interface`` for a model
    that is not going to be coupled.

    Parameters
    ----------
    mesh
        Quadrilateral mesh of this piece.
    material
        Isotropic plane-stress constants.
    interface
        Boundary or boundaries kept as the interface block. ``None`` leaves
        that block empty.
    clamp
        Boundary whose degrees of freedom are removed. ``None`` removes nothing.

    Returns
    -------
    Model
        Free stiffness and mass, with the interface index into those arrays.
    """
    stiffness, mass, basis = assemble_plane_stress(mesh, material)
    boundary = _interface_dofs(basis, interface)
    clamped = (
        np.array([], dtype=np.int64)
        if clamp is None
        else _ordered_boundary(basis, clamp)
    )
    if np.intersect1d(boundary, clamped).size:
        raise ValueError("The clamp and the interface share degrees of freedom")
    return _eliminate(stiffness, mass, basis, boundary, clamped)


def vibration_modes(model: Model, n_modes: int) -> ModeShape:
    r"""Lowest mass-normalised vibration modes of a restrained model.

    The columns of the returned shape are the eigenvectors, ordered by
    increasing frequency, with :math:`\Phi^\mathsf{T} M \Phi = I`. Frequencies
    are :math:`\sqrt{\lambda}/(2\pi)` in hertz. Every eigenvalue has to be
    positive, so the model must already have its rigid motion removed.

    Parameters
    ----------
    model
        Restrained component. Its stiffness must be positive definite.
    n_modes
        Number of modes to compute, starting from the lowest frequency.

    Returns
    -------
    ModeShape
        Eigenvectors in ``values`` and frequencies in hertz in ``frequencies``.
    """
    eigenvalues, vectors = eigh(
        model.stiffness, model.mass, subset_by_index=[0, n_modes - 1]
    )
    frequencies = np.sqrt(eigenvalues) / (2.0 * np.pi)
    return ModeShape("vibration", vectors, frequencies)


def _interface_dofs(
    basis: Basis, interface: str | tuple[str, ...] | None
) -> NDArray[np.int64]:
    if interface is None:
        return np.array([], dtype=np.int64)
    names = (interface,) if isinstance(interface, str) else interface
    return np.concatenate([_ordered_boundary(basis, name) for name in names])


def _ordered_boundary(basis: Basis, name: str) -> NDArray[np.int64]:
    r"""Interface unknowns ordered by component, then by increasing :math:`y`.

    Parameters
    ----------
    basis
        Quadratic vector basis of the component.
    name
        Boundary name, one of ``left``, ``right``, ``bottom``, or ``top``.

    Returns
    -------
    ndarray
        Horizontal unknowns in order of increasing :math:`y`, followed by the
        vertical unknowns in the same order.
    """
    blocks = []
    for component in ("u^1", "u^2"):
        dofs = basis.get_dofs(name).all(component)
        order = np.argsort(basis.doflocs[1, dofs], kind="mergesort")
        blocks.append(dofs[order])
    return np.concatenate(blocks).astype(np.int64, copy=False)


def _component_ids(basis: Basis) -> NDArray[np.int64]:
    """0 for the horizontal unknown and 1 for the vertical unknown.

    Parameters
    ----------
    basis
        Quadratic vector basis whose degrees of freedom are numbered.

    Returns
    -------
    ndarray
        Component index of every degree of freedom, ``0`` then ``1``.
    """
    component = np.empty(basis.dofs.N, dtype=np.int64)
    for block in (
        basis.dofs.nodal_dofs,
        basis.dofs.facet_dofs,
        basis.dofs.edge_dofs,
        basis.dofs.interior_dofs,
    ):
        if block.size == 0:
            continue
        for row in range(block.shape[0]):
            component[block[row]] = row % basis.dofs.element.dim
    return component


def _eliminate(stiffness, mass, basis: Basis, boundary, clamped) -> Model:
    """Drop the clamped unknowns and renumber what remains.

    Parameters
    ----------
    stiffness : ndarray
        Stiffness on the full basis, including clamped degrees of freedom.
    mass : ndarray
        Mass in the same ordering as ``stiffness``.
    basis
        Quadratic vector basis that numbers those degrees of freedom.
    boundary : ndarray
        Interface indices into the full basis.
    clamped : ndarray
        Indices of the prescribed unknowns to remove.

    Returns
    -------
    Model
        Free stiffness and mass. ``boundary`` is remapped into that numbering,
        and a clamped vertex unknown is stored as ``-1``.
    """
    keep = np.ones(stiffness.shape[0], dtype=bool)
    keep[clamped] = False
    free = np.flatnonzero(keep)
    remap = np.full(stiffness.shape[0], -1, dtype=np.int64)
    remap[free] = np.arange(free.size)

    nodal = basis.dofs.nodal_dofs
    vertex_dofs = np.full((nodal.shape[1], nodal.shape[0]), -1, dtype=np.int64)

    for component in range(nodal.shape[0]):
        vertex_dofs[:, component] = remap[nodal[component]]

    return Model(
        stiffness=_symmetrize(stiffness[np.ix_(free, free)]),
        mass=_symmetrize(mass[np.ix_(free, free)]),
        boundary=remap[boundary],
        coordinates=basis.doflocs[:, free].T.copy(),
        component=_component_ids(basis)[free],
        vertex_dofs=vertex_dofs,
    )


def _symmetrize(matrix: NDArray[np.float64]) -> NDArray[np.float64]:
    r"""Replace a square matrix by its symmetric part.

    Parameters
    ----------
    matrix
        Square matrix, as returned by the finite-element assembly.

    Returns
    -------
    ndarray
        :math:`(A + A^\mathsf{T})/2`.
    """
    return 0.5 * (matrix + matrix.T)
