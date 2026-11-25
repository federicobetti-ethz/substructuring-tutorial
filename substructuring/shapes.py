"""Shapes returned by a component reduction."""

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
class ComponentReduction:
    """Reduced matrices of one substructure, still in its own coordinates.

    ``basis`` maps the reduced coordinates back to the substructure's free
    finite-element degrees of freedom, :math:`u = R q`. The leading columns
    are the boundary block.

    Attributes
    ----------
    stiffness
        Reduced stiffness :math:`R^\\mathsf{T} K R`.
    mass
        Reduced mass :math:`R^\\mathsf{T} M R`.
    basis
        Reduction basis :math:`R`.
    n_boundary
        Number of boundary unknowns, the leading columns of ``basis``.
    shapes
        Families of physical component shapes used for plotting.
    """

    stiffness: NDArray[np.float64]
    mass: NDArray[np.float64]
    basis: NDArray[np.float64]
    n_boundary: int
    shapes: tuple[ModeShape, ...] = ()
