"""Plane-stress assembly on a quadratic quadrilateral mesh."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from skfem import Basis, BilinearForm, ElementQuad2, ElementVector, MeshLine, asm
from skfem.helpers import ddot, dot, sym_grad, trace
from skfem.mesh import Mesh


@dataclass(frozen=True)
class Material:
    """Isotropic plane-stress material. Displacements are in metres."""

    young: float
    poisson: float
    density: float
    thickness: float


@dataclass(frozen=True)
class Model:
    """Stiffness and mass with prescribed displacements already removed.

    ``boundary`` indexes the interface degrees of freedom that component-mode
    synthesis keeps. Both arrays are ordered by increasing ``y``, with every
    horizontal unknown before the vertical unknown at the same location, so two
    components that share a cut can be coupled entry by entry.
    """

    stiffness: NDArray[np.float64]
    mass: NDArray[np.float64]
    boundary: NDArray[np.int64]
    coordinates: NDArray[np.float64]
    component: NDArray[np.int64]
    vertex_dofs: NDArray[np.int64]


def rectangle_mesh(x0: float, x1: float, height: float, nx: int, ny: int) -> Mesh:
    """Tensor-product quadrilateral mesh of ``[x0, x1] x [0, height]``."""
    if nx < 1 or ny < 1:
        raise ValueError("nx and ny must be positive")
    horizontal = MeshLine(np.linspace(x0, x1, nx + 1))
    vertical = MeshLine(np.linspace(0.0, height, ny + 1))
    return (horizontal * vertical).with_defaults()


def assemble_plane_stress(
    mesh: Mesh, material: Material
) -> tuple[NDArray[np.float64], NDArray[np.float64], Basis]:
    """Return dense stiffness, dense mass, and the quadratic vector basis.

    The constitutive law is plane stress. Both matrices are scaled by the
    thickness, so they are the stiffness and mass of a flat strip.
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
    """
    stiffness, mass, basis = assemble_plane_stress(mesh, material)
    boundary = _interface_dofs(basis, interface)
    clamped = np.array([], dtype=np.int64) if clamp is None else _ordered_boundary(basis, clamp)
    if np.intersect1d(boundary, clamped).size:
        raise ValueError("The clamp and the interface share degrees of freedom")
    return _eliminate(stiffness, mass, basis, boundary, clamped)


def _interface_dofs(basis: Basis, interface: str | tuple[str, ...] | None) -> NDArray[np.int64]:
    if interface is None:
        return np.array([], dtype=np.int64)
    names = (interface,) if isinstance(interface, str) else interface
    return np.concatenate([_ordered_boundary(basis, name) for name in names])


def _ordered_boundary(basis: Basis, name: str) -> NDArray[np.int64]:
    """Interface unknowns ordered by ``(component, y)``."""
    blocks = []
    for component in ("u^1", "u^2"):
        dofs = basis.get_dofs(name).all(component)
        order = np.argsort(basis.doflocs[1, dofs], kind="mergesort")
        blocks.append(dofs[order])
    return np.concatenate(blocks).astype(np.int64, copy=False)


def _component_ids(basis: Basis) -> NDArray[np.int64]:
    """0 for the horizontal unknown and 1 for the vertical unknown."""
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
    keep = np.ones(stiffness.shape[0], dtype=bool)
    keep[clamped] = False
    free = np.flatnonzero(keep)
    remap = np.full(stiffness.shape[0], -1, dtype=np.int64)
    remap[free] = np.arange(free.size)
    if np.any(remap[boundary] < 0):
        raise ValueError("An interface degree of freedom is clamped")
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
    return 0.5 * (matrix + matrix.T)
