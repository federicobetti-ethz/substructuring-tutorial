"""One substructure, with local and global degree-of-freedom numbers."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from substructuring.shapes import ComponentReduction, ModeShape


@dataclass(frozen=True)
class Substructure:
    """Stiffness and mass of one piece, numbered locally and globally.

    A local number is the row of ``stiffness``. A global number is the column
    of the localization :math:`L`: two boundary unknowns that share a global
    number are the same cut degree of freedom. Interior numbers are unique.

    Attributes
    ----------
    stiffness
        Matrix of this piece in local order.
    mass
        Mass matrix in the same order.
    interior_local
        Local numbers of the unknowns that belong only to this piece.
    boundary_local
        Local numbers of the cut unknowns, in the order stored by the basis.
    interior_global
        Global numbers of ``interior_local``, in the same order.
    boundary_global
        Global numbers of ``boundary_local``, in the same order.
    basis
        Map from these coordinates back to the parent finite-element
        displacement, :math:`u = R q`. The identity when this piece has not
        been reduced.
    shapes
        Families of physical shapes kept for plotting.
    """

    stiffness: NDArray[np.float64]
    mass: NDArray[np.float64]
    interior_local: NDArray[np.int64]
    boundary_local: NDArray[np.int64]
    interior_global: NDArray[np.int64]
    boundary_global: NDArray[np.int64]
    basis: NDArray[np.float64]
    shapes: tuple[ModeShape, ...] = ()

    def __post_init__(self) -> None:
        n_dof = self.stiffness.shape[0]
        if self.mass.shape != self.stiffness.shape:
            raise ValueError("Mass and stiffness have different shapes")
        if self.basis.shape[1] != n_dof:
            raise ValueError("The basis does not match the local coordinates")
        _check_partition(n_dof, self.interior_local, self.boundary_local, "local")
        if self.interior_local.shape != self.interior_global.shape:
            raise ValueError("Interior local and global numbers differ in length")
        if self.boundary_local.shape != self.boundary_global.shape:
            raise ValueError("Boundary local and global numbers differ in length")
        owned = np.concatenate([self.interior_global, self.boundary_global])
        if owned.size != np.unique(owned).size:
            raise ValueError("A substructure repeats one of its global numbers")

    @property
    def n_dof(self) -> int:
        """Number of local degrees of freedom."""
        return int(self.stiffness.shape[0])

    @property
    def next_free_global(self) -> int:
        """First global number not used by this substructure."""
        owned = np.concatenate([self.interior_global, self.boundary_global])
        if owned.size == 0:
            return 0
        return int(owned.max()) + 1

    def reduce(
        self,
        method: str,
        *,
        n_modes: int = 0,
        n_rigid: int = 0,
        first_global: int | None = None,
    ) -> "Substructure":
        """Replace the interior by Guyan, Craig–Bampton, or Rubin coordinates.

        The boundary keeps its global numbers. Each new modal coordinate
        receives the next global number, starting at ``first_global``.

        Parameters
        ----------
        method
            ``"guyan"``, ``"craig-bampton"``, or ``"rubin"``.
        n_modes
            Elastic modes to keep. Guyan reduction accepts only zero.
        n_rigid
            Rigid-body modes kept by Rubin before the elastic ones. Craig–Bampton
            and Guyan accept only zero.
        first_global
            Global number of the first new modal coordinate. The default is
            one past the numbers already used by this piece.

        Returns
        -------
        Substructure
            Reduced matrices. The basis still maps them to the original
            finite-element displacement.
        """
        from substructuring.craig_bampton import craig_bampton
        from substructuring.rubin import rubin
        from substructuring.static_condensation import static_condensation

        if method == "guyan":
            reduced = static_condensation(
                self.stiffness, self.mass, self.boundary_local
            )
        elif method == "craig-bampton":
            reduced = craig_bampton(
                self.stiffness, self.mass, self.boundary_local, n_modes
            )
        elif method == "rubin":
            reduced = rubin(
                self.stiffness,
                self.mass,
                self.boundary_local,
                n_modes,
                n_rigid=n_rigid,
            )
        else:
            raise ValueError("method must be 'guyan', 'craig-bampton', or 'rubin'")
        return self._with_reduction(reduced, first_global)

    def _with_reduction(
        self, reduced: ComponentReduction, first_global: int | None
    ) -> "Substructure":
        n_modal = reduced.stiffness.shape[0] - reduced.n_boundary
        if first_global is None:
            first_global = self.next_free_global
        return Substructure(
            stiffness=reduced.stiffness,
            mass=reduced.mass,
            boundary_local=np.arange(reduced.n_boundary, dtype=np.int64),
            interior_local=np.arange(
                reduced.n_boundary, reduced.n_boundary + n_modal, dtype=np.int64
            ),
            boundary_global=np.array(self.boundary_global, dtype=np.int64, copy=True),
            interior_global=np.arange(
                first_global, first_global + n_modal, dtype=np.int64
            ),
            basis=reduced.basis,
            shapes=reduced.shapes,
        )


def _check_partition(
    n_dof: int,
    interior: NDArray[np.int64],
    boundary: NDArray[np.int64],
    kind: str,
) -> None:
    combined = np.concatenate([interior, boundary])
    expected = np.arange(n_dof)
    if combined.size != n_dof or not np.array_equal(np.sort(combined), expected):
        raise ValueError(f"The {kind} numbers must be each degree of freedom once")
