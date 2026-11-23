"""Primal assembly of reduced components that share one interface."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class ReducedModel:
    """One component in coordinates ``[interface displacements, local coordinates]``.

    ``basis`` maps those coordinates back to the component's free finite-element
    degrees of freedom: ``u = basis @ q``.
    """

    stiffness: NDArray[np.float64]
    mass: NDArray[np.float64]
    basis: NDArray[np.float64]
    n_boundary: int


@dataclass(frozen=True)
class CoupledModel:
    """Two or more reduced components coupled by shared interface displacements."""

    stiffness: NDArray[np.float64]
    mass: NDArray[np.float64]
    n_boundary: int
    local: tuple[NDArray[np.int64], ...]


def couple(models: tuple[ReducedModel, ...] | list[ReducedModel]) -> CoupledModel:
    """Assemble reduced components whose interface blocks are in the same order."""
    if len(models) < 2:
        raise ValueError("At least two components are required")
    n_boundary = models[0].n_boundary
    local_sizes = []
    for model in models:
        if model.n_boundary != n_boundary:
            raise ValueError("Every component must use the same interface ordering")
        local_sizes.append(model.stiffness.shape[0] - n_boundary)
        if local_sizes[-1] < 0:
            raise ValueError("A reduced model is smaller than its interface")

    size = n_boundary + sum(local_sizes)
    stiffness = np.zeros((size, size))
    mass = np.zeros((size, size))
    local = []
    cursor = n_boundary
    for model, n_local in zip(models, local_sizes):
        slots = np.concatenate(
            [np.arange(n_boundary), np.arange(cursor, cursor + n_local)]
        )
        stiffness[np.ix_(slots, slots)] += model.stiffness
        mass[np.ix_(slots, slots)] += model.mass
        local.append(np.arange(cursor, cursor + n_local))
        cursor += n_local
    return CoupledModel(
        stiffness=0.5 * (stiffness + stiffness.T),
        mass=0.5 * (mass + mass.T),
        n_boundary=n_boundary,
        local=tuple(local),
    )
