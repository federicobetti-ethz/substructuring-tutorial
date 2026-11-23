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
    """

    label: str
    values: NDArray[np.float64]
    frequencies: NDArray[np.float64] | None = None


@dataclass(frozen=True)
class ReducedModel:
    """One component in coordinates ``[interface displacements, local coordinates]``.

    ``basis`` maps those coordinates back to the component's free finite-element
    degrees of freedom: ``u = basis @ q``. ``shapes`` holds the same information
    split into the families one wants to draw.
    """

    stiffness: NDArray[np.float64]
    mass: NDArray[np.float64]
    basis: NDArray[np.float64]
    n_boundary: int
    shapes: tuple[ModeShape, ...] = ()


@dataclass(frozen=True)
class CoupledModel:
    """Reduced components coupled along a chain of interfaces.

    ``boundary[i]`` holds the global positions of component ``i``'s interface
    coordinates. ``local[i]`` holds its modal coordinates. The global vector is
    ordered as every cut, from the clamped end toward the tip, followed by the
    modal coordinates of each component in that same order.
    """

    stiffness: NDArray[np.float64]
    mass: NDArray[np.float64]
    boundary: tuple[NDArray[np.int64], ...]
    local: tuple[NDArray[np.int64], ...]


def couple(models: tuple[ReducedModel, ...] | list[ReducedModel]) -> CoupledModel:
    """Assemble components ordered from the clamped end to the free end.

    The first and last components carry one interface block. Each interior
    component carries the left cut followed by the right cut. Cuts that meet
    are required to have the same number of degrees of freedom, in the same
    order.
    """
    components = list(models)
    count = len(components)
    if count < 2:
        raise ValueError("At least two components are required")
    cut_sizes = _cut_sizes(components)
    offsets = np.cumsum([0, *cut_sizes[:-1]])
    n_interface = int(sum(cut_sizes))
    local_sizes = [component.stiffness.shape[0] - component.n_boundary for component in components]
    if any(size < 0 for size in local_sizes):
        raise ValueError("A reduced model is smaller than its interface")

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
    sizes = [components[0].n_boundary]
    for index in range(1, len(components) - 1):
        left = sizes[-1]
        width = components[index].n_boundary - left
        if width <= 0:
            raise ValueError(
                f"Component {index} must store the left cut before the right cut"
            )
        sizes.append(width)
    if components[-1].n_boundary != sizes[-1]:
        raise ValueError("The last component's interface does not match the previous cut")
    return sizes


def _component_cuts(index: int, count: int) -> list[int]:
    if index == 0:
        return [0]
    if index == count - 1:
        return [count - 2]
    return [index - 1, index]
