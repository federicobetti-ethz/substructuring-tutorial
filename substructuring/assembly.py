"""Primal assembly through the localization matrix."""

from collections import Counter
from dataclasses import dataclass, replace

import numpy as np
from numpy.typing import NDArray
from scipy.linalg import block_diag

from substructuring.substructure import Substructure


@dataclass(frozen=True)
class PrimalAssembly:
    """Coupled matrices :math:`L^\\mathsf{T} K L` and :math:`L^\\mathsf{T} M L`.

    Attributes
    ----------
    stiffness
        Primal stiffness.
    mass
        Primal mass.
    localization
        Boolean matrix :math:`L` such that the stacked displacement is
        :math:`u = L q`.
    constraint
        Signed Boolean matrix :math:`B` of the cuts, with :math:`B L = 0`.
    """

    stiffness: NDArray[np.float64]
    mass: NDArray[np.float64]
    localization: NDArray[np.float64]
    constraint: NDArray[np.float64]


class GlobalModel:
    """Substructures tied by shared global numbers on the cuts.

    The stacked displacement lists every local unknown of every piece, so a
    cut appears twice. :meth:`localization` builds the Boolean matrix :math:`L`
    whose columns are the unique global numbers. :meth:`constraint` builds the
    signed matrix :math:`B` with one row for each matched pair. These two
    matrices satisfy :math:`B L = 0`, and therefore

    .. math::

       L^\\mathsf{T} g = 0

    for every connecting force :math:`g = B^\\mathsf{T} \\lambda`.
    """

    def __init__(self, substructures: tuple[Substructure, ...] | list[Substructure]):
        self.substructures = tuple(substructures)
        if len(self.substructures) < 1:
            raise ValueError("A global model needs at least one substructure")
        self._check_global_numbers()

    @property
    def n_global(self) -> int:
        """Number of unique global degrees of freedom."""
        return int(self._global_ids().size)

    @property
    def next_free_global(self) -> int:
        """First global number not used by any substructure."""
        return self.n_global

    def localization(self) -> NDArray[np.float64]:
        """Boolean :math:`L` with one column per global degree of freedom.

        The column of a global number contains a one at every local unknown
        that carries that number. Both sides of a cut therefore receive the
        same displacement, :math:`u = L q`.
        """
        rows = sum(sub.n_dof for sub in self.substructures)
        localization = np.zeros((rows, self.n_global))
        offset = 0
        for sub in self.substructures:
            for local, global_index in _pairs(sub.boundary_local, sub.boundary_global):
                localization[offset + local, global_index] = 1.0
            for local, global_index in _pairs(sub.interior_local, sub.interior_global):
                localization[offset + local, global_index] = 1.0
            offset += sub.n_dof
        return localization

    def constraint(self) -> NDArray[np.float64]:
        """Signed Boolean :math:`B` of the cuts.

        Each row is ``+1`` and ``-1`` on the two local unknowns that share a
        global number. A stacked displacement is compatible when
        :math:`B u = 0`.
        """
        positions: dict[int, list[int]] = {}
        offset = 0
        for sub in self.substructures:
            for local, global_index in _pairs(sub.boundary_local, sub.boundary_global):
                positions.setdefault(global_index, []).append(offset + local)
            offset += sub.n_dof
        matched = [pair for pair in positions.values() if len(pair) == 2]
        constraint = np.zeros((len(matched), offset))
        for row, (first, second) in enumerate(matched):
            constraint[row, first] = 1.0
            constraint[row, second] = -1.0
        return constraint

    def stacked_stiffness(self) -> NDArray[np.float64]:
        """Block-diagonal stack :math:`\\operatorname{diag}(K^{(s)})`."""
        return block_diag(*(sub.stiffness for sub in self.substructures))

    def stacked_mass(self) -> NDArray[np.float64]:
        """Block-diagonal stack :math:`\\operatorname{diag}(M^{(s)})`."""
        return block_diag(*(sub.mass for sub in self.substructures))

    def assemble(self) -> PrimalAssembly:
        """Form the primal products :math:`L^\\mathsf{T} K L` and :math:`L^\\mathsf{T} M L`.

        :math:`K` and :math:`M` are the block-diagonal stacks. The product is
        checked against the cut constraints: :math:`B L` has to vanish, which
        is the statement :math:`L^\\mathsf{T} g = 0` for a connecting force
        :math:`g = B^\\mathsf{T} \\lambda`.
        """
        localization = self.localization()
        constraint = self.constraint()
        if constraint.size and not np.allclose(
            constraint @ localization, 0.0, atol=1e-8
        ):
            raise RuntimeError("L is not in the kernel of the cut constraints")
        stiffness = self.stacked_stiffness()
        mass = self.stacked_mass()
        return PrimalAssembly(
            stiffness=localization.T @ stiffness @ localization,
            mass=localization.T @ mass @ localization,
            localization=localization,
            constraint=constraint,
        )

    def reduce(
        self,
        method: str,
        *,
        n_modes: int | tuple[int, ...] | list[int] = 0,
        n_rigid: int | tuple[int, ...] | list[int] = 0,
    ) -> "GlobalModel":
        """Reduce every substructure, keeping the global numbers of each cut.

        Modal coordinates receive new global numbers, so the same localization
        couples the reduced blocks.

        Parameters
        ----------
        method
            ``"guyan"``, ``"craig-bampton"``, or ``"rubin"``.
        n_modes
            Elastic modes in each piece, or one integer shared by every piece.
        n_rigid
            Rigid-body modes in each piece, or one integer shared by every piece.

        Returns
        -------
        GlobalModel
            One reduced substructure per input piece, in the same order.
        """
        modes = _per_piece(n_modes, len(self.substructures), "n_modes")
        rigid = _per_piece(n_rigid, len(self.substructures), "n_rigid")
        cursor = self.next_free_global
        reduced = []
        for sub, n_mode, n_body in zip(self.substructures, modes, rigid):
            piece = sub.reduce(
                method, n_modes=n_mode, n_rigid=n_body, first_global=cursor
            )
            cursor = max(cursor, piece.next_free_global)
            reduced.append(piece)
        return GlobalModel(_compact(reduced))

    def project_force(
        self, index: int, force: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        """Carry a force on one piece into the primal coordinates.

        The basis takes the finite-element force into the substructure
        coordinates, and :math:`L^\\mathsf{T}` sums the two sides of each cut.
        """
        sub = self.substructures[index]
        local = sub.basis.T @ force
        stacked = np.zeros(sum(item.n_dof for item in self.substructures))
        offset = sum(item.n_dof for item in self.substructures[:index])
        stacked[offset : offset + sub.n_dof] = local
        return self.localization().T @ stacked

    def recovery_row(self, index: int, dof: int) -> NDArray[np.float64]:
        """Row that reads one finite-element displacement from the primal vector.

        The piece's coordinates are the rows of :math:`L` that belong to it,
        and the finite-element displacement is :math:`R` times those coordinates.
        """
        sub = self.substructures[index]
        offset = sum(item.n_dof for item in self.substructures[:index])
        block = self.localization()[offset : offset + sub.n_dof]
        return sub.basis[dof] @ block

    def _check_global_numbers(self) -> None:
        interior = [
            int(number) for sub in self.substructures for number in sub.interior_global
        ]
        boundary = [
            int(number) for sub in self.substructures for number in sub.boundary_global
        ]
        if len(interior) != len(set(interior)):
            raise ValueError("An interior degree of freedom is used twice")
        if set(interior) & set(boundary):
            raise ValueError("A degree of freedom is both interior and on a cut")
        unique = set(interior) | set(boundary)
        if unique != set(range(len(unique))):
            raise ValueError("Global numbers must be 0, 1, ..., n-1 without gaps")
        for number, count in Counter(boundary).items():
            if count > 2:
                raise ValueError(
                    f"Global degree of freedom {number} meets more than two pieces"
                )

    def _global_ids(self) -> NDArray[np.int64]:
        parts = [
            np.concatenate([sub.interior_global, sub.boundary_global])
            for sub in self.substructures
        ]
        if not parts or all(part.size == 0 for part in parts):
            return np.array([], dtype=np.int64)
        return np.unique(np.concatenate(parts))


def chain(models) -> GlobalModel:
    """Number a chain of finite-element components and return the global model.

    Components are ordered from the clamped end to the free end. The first
    boundary block is one cut. An interior component stores the upstream cut
    followed by the downstream cut. The last boundary block repeats the
    upstream cut. Interior unknowns receive global numbers of their own.

    Parameters
    ----------
    models
        Finite-element components. Each exposes ``stiffness``, ``mass``, and
        ``boundary``.

    Returns
    -------
    GlobalModel
        One substructure per component. Matching degrees of freedom on a cut
        share a global number.
    """
    if len(models) < 2:
        raise ValueError("A chain needs at least two components")
    substructures = []
    next_id = 0
    upstream: NDArray[np.int64] | None = None
    for index, model in enumerate(models):
        boundary_local = np.array(model.boundary, dtype=np.int64, copy=True)
        interior_local = np.setdiff1d(
            np.arange(model.stiffness.shape[0]), boundary_local, assume_unique=False
        )
        if index == 0:
            boundary_global = np.arange(next_id, next_id + boundary_local.size)
            next_id += boundary_local.size
            upstream = boundary_global
        elif index == len(models) - 1:
            if upstream is None or upstream.size != boundary_local.size:
                raise ValueError(
                    "The last component's cut does not match the previous one"
                )
            boundary_global = np.array(upstream, dtype=np.int64, copy=True)
        else:
            if upstream is None or boundary_local.size <= upstream.size:
                raise ValueError(
                    "An interior component must store the upstream cut, then the downstream cut"
                )
            downstream = np.arange(
                next_id, next_id + boundary_local.size - upstream.size
            )
            next_id += downstream.size
            boundary_global = np.concatenate([upstream, downstream])
            upstream = downstream
        interior_global = np.arange(next_id, next_id + interior_local.size)
        next_id += interior_local.size
        substructures.append(
            Substructure(
                stiffness=model.stiffness,
                mass=model.mass,
                interior_local=interior_local.astype(np.int64, copy=False),
                boundary_local=boundary_local,
                interior_global=interior_global,
                boundary_global=boundary_global.astype(np.int64, copy=False),
                basis=np.eye(model.stiffness.shape[0]),
            )
        )
    return GlobalModel(substructures)


def _compact(substructures: list[Substructure]) -> list[Substructure]:
    """Renumber global degrees of freedom as 0, 1, ..., n-1 without gaps.

    Shared boundary numbers stay shared. The order is the order in which the
    numbers first appear, piece by piece, boundary before the new coordinates.
    """
    order: list[int] = []
    seen: set[int] = set()
    for sub in substructures:
        numbers = np.concatenate([sub.boundary_global, sub.interior_global])
        for number in numbers.tolist():
            if number not in seen:
                seen.add(number)
                order.append(number)
    remap = {old: new for new, old in enumerate(order)}
    compacted = []
    for sub in substructures:
        compacted.append(
            replace(
                sub,
                boundary_global=np.array(
                    [remap[int(number)] for number in sub.boundary_global],
                    dtype=np.int64,
                ),
                interior_global=np.array(
                    [remap[int(number)] for number in sub.interior_global],
                    dtype=np.int64,
                ),
            )
        )
    return compacted


def _pairs(local: NDArray[np.int64], global_index: NDArray[np.int64]):
    return zip(local.tolist(), global_index.tolist())


def _per_piece(
    value: int | tuple[int, ...] | list[int], count: int, name: str
) -> list[int]:
    if isinstance(value, int):
        return [value] * count
    values = [int(item) for item in value]
    if len(values) != count:
        raise ValueError(f"{name} needs one entry per substructure")
    return values
