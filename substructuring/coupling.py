"""Primal assembly of a chain of reduced components."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class ModeShape:
    """Displacements of the free degrees of freedom.

    Each column of ``values`` is one shape. ``frequencies`` is in hertz and is
    omitted for static shapes. These are the physical component shapes: Rubin
    free-interface modes still carry their interface motion here, even though
    the reduction basis moves that motion into the attachment modes.

    Attributes
    ----------
    label
        Name of this family of shapes, such as ``"constraint"`` or
        ``"free interface"``.
    values
        One displacement column per shape, on the free finite-element
        unknowns.
    frequencies
        Natural frequencies in hertz, one per column of ``values``. ``None``
        for a static family.
    """

    label: str
    values: NDArray[np.float64]
    frequencies: NDArray[np.float64] | None = None


@dataclass(frozen=True)
class ReducedModel:
    r"""One component in coordinates :math:`(u_b, \eta)`.

    ``basis`` maps those coordinates back to the component's free finite-element
    degrees of freedom, :math:`u = R q`. ``shapes`` holds the same information
    split into the families one wants to draw.

    Attributes
    ----------
    stiffness
        Reduced stiffness :math:`R^\mathsf{T} K R`.
    mass
        Reduced mass :math:`R^\mathsf{T} M R`.
    basis
        Reduction basis :math:`R`. Columns are ordered as the interface block
        followed by the local coordinates.
    n_boundary
        Number of interface unknowns, the leading columns of ``basis``.
    shapes
        Families of physical component shapes used for plotting.
    """

    stiffness: NDArray[np.float64]
    mass: NDArray[np.float64]
    basis: NDArray[np.float64]
    n_boundary: int
    shapes: tuple[ModeShape, ...] = ()


@dataclass(frozen=True)
class CoupledModel:
    r"""Reduced components coupled along a chain of interfaces.

    ``boundary[i]`` holds the global positions of component ``i``'s interface
    coordinates. ``local[i]`` holds its modal coordinates. The global vector is
    ordered as every cut, from the clamped end toward the tip, followed by the
    modal coordinates of each component in that same order,

    .. math::

       q = \bigl(u^{(0)}, \ldots, u^{(N-2)},\;
       \eta^{(0)}, \ldots, \eta^{(N-1)}\bigr).

    Attributes
    ----------
    stiffness
        Coupled stiffness in the coordinates :math:`q`.
    mass
        Coupled mass in the same coordinates.
    boundary
        For each component, the indices of its interface unknowns in :math:`q`.
    local
        For each component, the indices of its local coordinates in :math:`q`.
    """

    stiffness: NDArray[np.float64]
    mass: NDArray[np.float64]
    boundary: tuple[NDArray[np.int64], ...]
    local: tuple[NDArray[np.int64], ...]


def couple(models: tuple[ReducedModel, ...] | list[ReducedModel]) -> CoupledModel:
    r"""Assemble components ordered from the clamped end to the free end.

    The first and last components carry one interface block. Each interior
    component carries the left cut followed by the right cut. Cuts that meet
    are required to have the same number of degrees of freedom, in the same
    order. Each reduced matrix is added into the rows and columns of the cuts
    it touches and of its own modal coordinates. Shared cut displacements
    enforce continuity of the chain.

    Parameters
    ----------
    models
        Reduced components, ordered from the clamped end to the free end.
        At least two are required.

    Returns
    -------
    CoupledModel
        Primal assembly of the chain.
    """
    components = list(models)
    count = len(components)

    cut_sizes = _cut_sizes(components)
    offsets = np.cumsum([0, *cut_sizes[:-1]])
    n_interface = int(sum(cut_sizes))
    local_sizes = [
        component.stiffness.shape[0] - component.n_boundary for component in components
    ]

    size = n_interface + sum(local_sizes)
    stiffness = np.zeros((size, size))
    mass = np.zeros((size, size))
    boundary_maps = []
    local = []
    cursor = n_interface

    for index, component in enumerate(components):
        cuts = _component_cuts(index, count)
        boundary_index = np.concatenate(
            [np.arange(offsets[cut], offsets[cut] + cut_sizes[cut]) for cut in cuts]
        )
        local_index = np.arange(cursor, cursor + local_sizes[index])
        slots = np.concatenate([boundary_index, local_index])
        stiffness[np.ix_(slots, slots)] += component.stiffness
        mass[np.ix_(slots, slots)] += component.mass
        boundary_maps.append(boundary_index.astype(np.int64, copy=False))
        local.append(local_index)
        cursor += local_sizes[index]

    return CoupledModel(
        stiffness=0.5 * (stiffness + stiffness.T),
        mass=0.5 * (mass + mass.T),
        boundary=tuple(boundary_maps),
        local=tuple(local),
    )


def _cut_sizes(components: list[ReducedModel]) -> list[int]:
    """Number of degrees of freedom on each cut of the chain.

    The root piece carries only the first cut, so its interface length is
    that cut. Each interior piece stores the left cut followed by the right
    cut, and the right-cut length is what remains of its interface. The tip
    piece repeats the last cut and does not add one.

    Parameters
    ----------
    components
        Reduced components, ordered from the clamped end to the free end.
        At least two are required.

    Returns
    -------
    list of int
        Width of each cut, from the wall toward the tip. One entry fewer
        than ``components``.
    """
    sizes = [components[0].n_boundary]
    for index in range(1, len(components) - 1):
        left = sizes[-1]
        width = components[index].n_boundary - left
        sizes.append(width)
    return sizes


def _component_cuts(index: int, count: int) -> list[int]:
    """Cut indices touched by one component.

    Parameters
    ----------
    index
        Position of the component, starting at 0 on the clamped end.
    count
        Number of components in the chain. At least two.

    Returns
    -------
    list of int
        The root returns ``[0]``. The tip returns ``[count - 2]``. An
        interior component returns its left cut and then its right cut,
        ``[index - 1, index]``.
    """
    if index == 0:
        return [0]
    if index == count - 1:
        return [count - 2]
    return [index - 1, index]
