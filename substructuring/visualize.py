"""Deformed-mesh plots of component modes."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import PolyCollection
from numpy.typing import NDArray

from substructuring.fem import Model


def rigid_body_modes(model: Model) -> NDArray[np.float64]:
    """Three planar rigid motions: horizontal, vertical, and rotation.

    The eigen-solver returns some other orthonormal basis of the same space.
    These coordinate directions are the ones to draw.
    """
    coordinates = model.coordinates
    center = coordinates.mean(axis=0)
    horizontal = model.component == 0
    vertical = model.component == 1
    shapes = np.zeros((coordinates.shape[0], 3))
    shapes[horizontal, 0] = 1.0
    shapes[vertical, 1] = 1.0
    shapes[horizontal, 2] = -(coordinates[horizontal, 1] - center[1])
    shapes[vertical, 2] = coordinates[vertical, 0] - center[0]
    return shapes


def mid_interface_column(model: Model, component: int) -> int:
    """Interface column whose node is nearest the middle of the cut.

    ``component`` is 0 for the horizontal unknown and 1 for the vertical one.
    Columns follow the ordering of ``model.boundary``.
    """
    columns = np.flatnonzero(model.component[model.boundary] == component)
    if columns.size == 0:
        raise ValueError(f"The interface has no unknown of component {component}")
    height = model.coordinates[model.boundary[columns], 1]
    middle = 0.5 * (height.min() + height.max())
    return int(columns[np.argmin(np.abs(height - middle))])


def plot_modes(substructures, path: Path | str, *, suptitle: str | None = None, deflection: float = 0.6):
    """Draw each substructure's modes on its own piece of the beam.

    ``substructures`` is a sequence of ``(name, mesh, model, entries)``. Each
    entry is a ``(title, displacement)`` pair on that substructure's free
    degrees of freedom. Columns are substructures. Every panel uses the same
    view of the whole beam: the active piece is deformed, and the other pieces
    stay in place as a gray outline.
    """
    substructures = list(substructures)
    meshes = [mesh for _, mesh, _, _ in substructures]
    n_columns = len(substructures)
    n_rows = max(len(entries) for *_, entries in substructures)
    xlim, ylim = _beam_limits(meshes, deflection)
    aspect = (xlim[1] - xlim[0]) / (ylim[1] - ylim[0])
    panel_height = 1.15
    panel_width = panel_height * aspect
    figure, axes = plt.subplots(
        n_rows,
        n_columns,
        figsize=(panel_width * n_columns, panel_height * n_rows + 0.8),
        squeeze=False,
        layout="constrained",
    )
    collections = []
    for column, (name, mesh, model, entries) in enumerate(substructures):
        named = False
        for row in range(n_rows):
            axis = axes[row, column]
            axis.set_axis_off()
            axis.set_aspect("equal", adjustable="box")
            for other in meshes:
                _draw_ghost(axis, other)
            if row >= len(entries) or entries[row] is None:
                axis.set_xlim(xlim)
                axis.set_ylim(ylim)
                continue
            title, displacement = entries[row]
            motion = _vertex_motion(model, displacement)
            drawn = _deformed_positions(mesh, motion, meshes, deflection)
            collections.append(_draw_fill(axis, mesh, drawn, motion))
            axis.set_xlim(xlim)
            axis.set_ylim(ylim)
            heading = f"{name}\n{title}" if not named else title
            named = True
            axis.set_title(heading, fontsize=9)
    if collections:
        color_scale = plt.Normalize(0.0, 1.0)
        for collection in collections:
            collection.set_norm(color_scale)
        figure.colorbar(
            collections[0],
            ax=axes,
            fraction=0.015,
            pad=0.01,
            label="Displacement / peak",
        )
    if suptitle:
        figure.suptitle(suptitle)
    output = Path(path)
    output.parent.mkdir(exist_ok=True)
    figure.savefig(output, dpi=150)
    plt.close(figure)
    return output


def _vertex_motion(model: Model, displacement: NDArray[np.float64]) -> NDArray[np.float64]:
    motion = np.zeros((model.vertex_dofs.shape[0], model.vertex_dofs.shape[1]))
    for component in range(model.vertex_dofs.shape[1]):
        index = model.vertex_dofs[:, component]
        active = index >= 0
        motion[active, component] = displacement[index[active]]
    return motion


def _beam_limits(meshes, deflection: float):
    points = np.hstack([mesh.p for mesh in meshes])
    x0, x1 = float(points[0].min()), float(points[0].max())
    y0, y1 = float(points[1].min()), float(points[1].max())
    pad = deflection * (y1 - y0)
    return (x0 - pad, x1 + pad), (y0 - pad, y1 + pad)


def _deformed_positions(mesh, motion, meshes, deflection: float):
    peak = np.max(np.linalg.norm(motion, axis=1))
    if peak == 0.0:
        return mesh.p.T
    height = max(float(np.ptp(item.p[1])) for item in meshes)
    return mesh.p.T + (deflection * height / peak) * motion


def _draw_ghost(axis, mesh) -> None:
    polygons = np.moveaxis(mesh.p.T[mesh.t], 0, 1)
    axis.add_collection(
        PolyCollection(
            polygons,
            facecolors="0.93",
            edgecolors="0.75",
            linewidths=0.4,
        )
    )


def _draw_fill(axis, mesh, positions, motion) -> PolyCollection:
    polygons = np.moveaxis(positions[mesh.t], 0, 1)
    magnitude = np.linalg.norm(motion, axis=1)
    color = magnitude[mesh.t].mean(axis=0)
    peak = color.max()
    if peak > 0.0:
        color = color / peak
    collection = PolyCollection(
        polygons,
        array=color,
        cmap="viridis",
        edgecolors="0.15",
        linewidths=0.7,
    )
    axis.add_collection(collection)
    axis.autoscale_view()
    return collection
