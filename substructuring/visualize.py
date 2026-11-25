"""Deformed-mesh plots of component modes."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.cm import ScalarMappable
from matplotlib.collections import PolyCollection
from matplotlib.lines import Line2D
from numpy.typing import NDArray

from substructuring.fem import Model, rectangle_mesh


def rigid_body_modes(model: Model) -> NDArray[np.float64]:
    """Three planar rigid motions: horizontal, vertical, and rotation.

    The eigen-solver returns some other orthonormal basis of the same space.
    These coordinate directions are the ones to draw.

    Parameters
    ----------
    model
        Free component. Both displacement components must be present.

    Returns
    -------
    ndarray
        Three columns: unit translation in :math:`x`, unit translation in
        :math:`y`, and rotation about the centroid of the free nodes.
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

    Columns follow the ordering of ``model.boundary``.

    Parameters
    ----------
    model
        Component whose interface block is searched.
    component
        ``0`` for the horizontal unknown and ``1`` for the vertical one.

    Returns
    -------
    int
        Column of ``model.boundary`` at the node closest to mid-height.
    """
    columns = np.flatnonzero(model.component[model.boundary] == component)
    if columns.size == 0:
        raise ValueError(f"The interface has no unknown of component {component}")
    height = model.coordinates[model.boundary[columns], 1]
    middle = 0.5 * (height.min() + height.max())
    return int(columns[np.argmin(np.abs(height - middle))])


def plot_modes(substructures, directory: Path | str, *, deflection: float = 0.6) -> list[Path]:
    """Write one figure per mode, containing every substructure.

    Every panel uses the same view of the whole beam. The active piece is
    deformed, and the other pieces stay in place as a gray outline. Colour is
    the displacement magnitude divided by its peak.

    Parameters
    ----------
    substructures : sequence of tuple
        Each item is ``(name, mesh, model, entries)``. ``entries`` lines up
        across substructures, so one file is one row. An entry is a
        ``(title, displacement)`` pair on that substructure's free degrees of
        freedom, or ``None`` when that piece has no shape in this mode.
    directory
        Directory where the figures are written. It is created if missing.
    deflection
        Amplitude of the drawn motion as a fraction of the beam height.

    Returns
    -------
    list of Path
        One path per written figure, in mode order.
    """
    substructures = list(substructures)
    meshes = [mesh for _, mesh, _, _ in substructures]
    n_modes = max(len(entries) for *_, entries in substructures)
    root = Path(directory)
    written = []
    for row in range(n_modes):
        panels = []
        titles = []
        for name, mesh, model, entries in substructures:
            entry = entries[row] if row < len(entries) else None
            panels.append((name, mesh, model, entry))
            if entry is not None:
                titles.append(entry[0])
        if not titles:
            continue
        path = root / f"{len(written) + 1:02d}_{_slug(' '.join(dict.fromkeys(titles)))}.png"
        _plot_mode(meshes, panels, path, deflection)
        written.append(path)
    return written


def plot_vibration_modes(mesh, model: Model, modes, directory: Path | str, *, deflection: float = 0.6) -> list[Path]:
    """Write one figure for each vibration mode of the full model.

    The undeformed mesh stays visible behind the deformed shape. Files are
    named from the mode frequencies, in hertz.

    Parameters
    ----------
    mesh : Mesh
        Quadrilateral mesh of the unreduced beam.
    model
        Free model of that mesh, with the wall already removed.
    modes : ModeShape
        Shape returned by :func:`substructuring.fem.vibration_modes`.
        ``frequencies`` has to be present.
    directory
        Directory where the figures are written. It is created if missing.
    deflection
        Amplitude of the drawn motion as a fraction of the beam height.

    Returns
    -------
    list of Path
        One path per mode, from the lowest frequency upward.
    """
    if modes.frequencies is None:
        raise ValueError("Vibration modes need frequencies")
    root = Path(directory)
    written = []
    for index, frequency in enumerate(modes.frequencies):
        title = f"{frequency:.0f} Hz"
        path = root / f"{index + 1:02d}_{_slug(title)}.png"
        displacement = modes.values[:, index]
        _plot_mode([mesh], [("", mesh, model, (title, displacement))], path, deflection)
        written.append(path)
    return written


def plot_static_condensation(
    path: Path | str,
    *,
    length: float = 2.0,
    height: float = 0.35,
    nx: int = 8,
    ny: int = 2,
    gap: float = 0.16,
) -> Path:
    """Draw which nodes static condensation retains.

    The beam is the same outline as the cantilever, meshed coarsely enough that
    every quadratic node is visible. Each piece keeps the nodes on the cut.
    The other free nodes are condensed. Nodes on the wall are prescribed, so
    they belong to neither set. The lower panel keeps only the retained nodes.

    Parameters
    ----------
    path
        Destination of the figure.
    length
        Length of the beam.
    height
        Height of the beam.
    nx
        Number of elements along the beam. It must be even, so the cut lies
        on element edges.
    ny
        Number of elements through the height.
    gap
        Distance by which the two pieces are pulled apart.

    Returns
    -------
    Path
        Written figure.
    """
    if nx % 2:
        raise ValueError("nx must be even so the cut lies on element edges")
    cut = 0.5 * length
    half = nx // 2
    left_mesh = rectangle_mesh(0.0, cut, height, half, ny)
    right_mesh = rectangle_mesh(cut, length, height, half, ny)
    left_nodes = _quadratic_nodes(0.0, cut, height, half, ny)
    right_nodes = _quadratic_nodes(cut, length, height, half, ny)
    left_sets = _node_sets(left_nodes, cut, clamp=True)
    right_sets = _node_sets(right_nodes, cut, clamp=False)
    y0, y1 = 0.0, height
    outer = 0.16 * height
    inner = 0.09 * height
    xlim = (-gap / 2 - 0.18, length + gap / 2 + 0.06)
    ylims = ((y0 - inner, y1 + outer), (y0 - outer, y1 + inner))
    figure, axes = plt.subplots(2, 1, figsize=(8.6, 3.35), layout="constrained")
    figure.get_layout_engine().set(h_pad=0.01, hspace=0.0)
    for axis, show_condensed, ylim in zip(axes, (True, False), ylims):
        _draw_condensation_panel(
            axis,
            left_mesh,
            right_mesh,
            left_sets,
            right_sets,
            shift=gap / 2,
            wall=-gap / 2,
            xlim=xlim,
            ylim=ylim,
            show_condensed=show_condensed,
        )
    retained = Line2D(
        [], [], linestyle="none", marker="o", markersize=6, markerfacecolor="0.1", markeredgecolor="0.1"
    )
    condensed = Line2D(
        [], [], linestyle="none", marker="o", markersize=6, markerfacecolor="white", markeredgecolor="0.15"
    )
    clamped = Line2D(
        [], [], linestyle="none", marker="s", markersize=5.5, markerfacecolor="0.45", markeredgecolor="0.45"
    )
    figure.legend(
        (retained, condensed, clamped),
        ("Retained", "Condensed", "Clamped"),
        loc="outside upper center",
        ncol=3,
        frameon=False,
    )
    output = Path(path)
    _save_mesh_figure(figure, output)
    return output


def plot_tip(mesh, path: Path | str) -> Path:
    r"""Draw the mesh and mark the upper free corner in red.

    That corner is the tip displacement reported by the cantilever comparison.

    Parameters
    ----------
    mesh : Mesh
        Quadrilateral mesh of the beam. The mark is the corner at the largest
        :math:`x` and the largest :math:`y`.
    path
        Destination of the figure.

    Returns
    -------
    Path
        Written figure.
    """
    y0 = float(mesh.p[1].min())
    y1 = float(mesh.p[1].max())
    x0 = float(mesh.p[0].min())
    x1 = float(mesh.p[0].max())
    pad_y = 0.55 * (y1 - y0)
    figure, axis = plt.subplots(figsize=(8.6, 2.5), layout="constrained")
    _draw_colored_elements(axis, mesh, np.ones(mesh.t.shape[1], dtype=bool), "#d9d9d9", 0.0)
    _style_mesh_axis(axis, (x0 - 0.18, x1 + 0.06), (y0 - pad_y, y1 + pad_y))
    _draw_wall(axis, x0, y0, y1)
    axis.scatter([x1], [y1], s=42, c="#d62728", zorder=6, clip_on=False)
    output = Path(path)
    _save_mesh_figure(figure, output)
    return output


def plot_model(mesh, path: Path | str) -> Path:
    """Draw the unreduced mesh, clamped at its left end.

    Parameters
    ----------
    mesh : Mesh
        Quadrilateral mesh of the beam. The wall is drawn at its left edge.
    path
        Destination of the figure.

    Returns
    -------
    Path
        Written figure.
    """
    y0 = float(mesh.p[1].min())
    y1 = float(mesh.p[1].max())
    x0 = float(mesh.p[0].min())
    x1 = float(mesh.p[0].max())
    pad_y = 0.55 * (y1 - y0)
    figure, axis = plt.subplots(figsize=(8.6, 2.5), layout="constrained")
    _draw_colored_elements(axis, mesh, np.ones(mesh.t.shape[1], dtype=bool), "#d9d9d9", 0.0)
    _style_mesh_axis(axis, (x0 - 0.18, x1 + 0.06), (y0 - pad_y, y1 + pad_y))
    _draw_wall(axis, x0, y0, y1)
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=150)
    plt.close(figure)
    return output


def plot_assembly(
    mesh,
    primal: Path | str,
    dual: Path | str,
    *,
    cut: float,
    gap: float = 0.16,
) -> tuple[Path, Path]:
    r"""Draw primal assembly and dual assembly as two figures.

    The two colors are the substructures on either side of ``cut``. The primal
    figure first shows the pieces apart, each with its own interface nodes,
    then joined so those nodes are one displacement unknown :math:`u`. The
    dual figure shows the pieces apart and then tied, each pair by a force
    unknown :math:`\lambda`.

    Parameters
    ----------
    mesh : Mesh
        Quadrilateral mesh of the unreduced beam. ``cut`` must lie on element
        edges.
    primal
        Path of the primal-assembly figure.
    dual
        Path of the dual-assembly figure.
    cut
        Abscissa of the interface.
    gap
        Distance by which the two pieces are pulled apart in a separated
        panel, in the same units as the mesh.

    Returns
    -------
    primal_path : Path
        Written primal figure.
    dual_path : Path
        Written dual figure.
    """
    centers = mesh.p[:, mesh.t].mean(axis=1)
    left = centers[0] < cut
    right = ~left
    interface = mesh.p[:, np.isclose(mesh.p[0], cut, atol=1e-8)]
    interface = interface[:, np.argsort(interface[1])]
    y0 = float(mesh.p[1].min())
    y1 = float(mesh.p[1].max())
    x0 = float(mesh.p[0].min())
    x1 = float(mesh.p[0].max())
    pad_y = 0.55 * (y1 - y0)
    xlim = (x0 - gap / 2 - 0.18, x1 + gap / 2 + 0.06)
    ylim = (y0 - pad_y, y1 + pad_y)

    primal_path = Path(primal)
    dual_path = Path(dual)
    primal_figure, primal_axes = plt.subplots(2, 1, figsize=(8.6, 5.2), layout="constrained")
    _draw_assembly_panel(
        primal_axes[0], mesh, left, right, interface, shift=gap / 2, wall=x0 - gap / 2, xlim=xlim, ylim=ylim
    )
    _draw_assembly_panel(
        primal_axes[1], mesh, left, right, interface, shift=0.0, wall=x0, xlim=xlim, ylim=ylim
    )
    _mark_interface(primal_axes[1], interface[0], interface[1], r"$u$")
    _save_mesh_figure(primal_figure, primal_path)

    dual_figure, dual_axes = plt.subplots(2, 1, figsize=(8.6, 5.2), layout="constrained")
    _draw_assembly_panel(
        dual_axes[0], mesh, left, right, interface, shift=gap / 2, wall=x0 - gap / 2, xlim=xlim, ylim=ylim
    )
    _draw_assembly_panel(
        dual_axes[1], mesh, left, right, interface, shift=gap / 2, wall=x0 - gap / 2, xlim=xlim, ylim=ylim
    )
    _draw_interface_forces(dual_axes[1], interface, gap)
    _mark_interface(dual_axes[1], np.full(interface.shape[1], cut), interface[1], r"$\lambda$")
    _save_mesh_figure(dual_figure, dual_path)
    return primal_path, dual_path


def _save_mesh_figure(figure, path: Path) -> None:
    """Write a mesh figure and close it.

    Parameters
    ----------
    figure : Figure
        Matplotlib figure to save.
    path
        Destination. Parent directories are created if missing.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=150)
    plt.close(figure)


def _plot_mode(meshes, panels, path: Path, deflection: float) -> None:
    """Draw one mode, with every substructure in its own panel.

    Each panel shows the whole beam. The active piece is deformed and the
    other pieces stay in place as a gray outline. Colour is the displacement
    magnitude divided by its peak.

    Parameters
    ----------
    meshes : sequence of Mesh
        Every substructure mesh. They set the common view and the gray outlines.
    panels : sequence of tuple
        Each item is ``(name, mesh, model, entry)``. ``entry`` is a
        ``(title, displacement)`` pair, or ``None`` when that piece has no
        shape in this mode.
    path
        Destination of the figure. Parent directories are created if missing.
    deflection
        Amplitude of the drawn motion as a fraction of the beam height.
    """
    xlim, ylim = _beam_limits(meshes, deflection)
    aspect = (xlim[1] - xlim[0]) / (ylim[1] - ylim[0])
    panel_height = 2.2
    panel_width = panel_height * aspect
    figure, axes = plt.subplots(
        1,
        len(panels),
        figsize=(panel_width * len(panels) + 1.15, panel_height + 0.45),
        squeeze=False,
        layout="constrained",
    )
    collections = []
    for axis, (name, mesh, model, entry) in zip(axes[0], panels):
        axis.set_axis_off()
        axis.set_aspect("equal", adjustable="box")
        for other in meshes:
            _draw_ghost(axis, other)
        axis.set_xlim(xlim)
        axis.set_ylim(ylim)
        if entry is None:
            continue
        title, displacement = entry
        motion = _vertex_motion(model, displacement)
        drawn = _deformed_positions(mesh, motion, meshes, deflection)
        collections.append(_draw_fill(axis, mesh, drawn, motion))
    if collections:
        color_scale = plt.Normalize(0.0, 1.0)
        for collection in collections:
            collection.set_norm(color_scale)
        colorbar = figure.colorbar(
            ScalarMappable(norm=color_scale, cmap="viridis"),
            ax=axes,
            fraction=0.03,
            pad=0.02,
        )
        colorbar.set_label("Displacement / peak")
        colorbar.set_ticks((0.0, 0.5, 1.0))
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=150)
    plt.close(figure)


def _quadratic_nodes(x0: float, x1: float, height: float, nx: int, ny: int) -> NDArray[np.float64]:
    r"""Nodes of a biquadratic tensor mesh, one row per node.

    Parameters
    ----------
    x0
        Left edge.
    x1
        Right edge.
    height
        Top edge. The bottom edge is :math:`y = 0`.
    nx
        Number of elements along :math:`x`.
    ny
        Number of elements along :math:`y`.

    Returns
    -------
    ndarray
        Node coordinates, shape ``((2 n_x + 1)(2 n_y + 1), 2)``.
    """
    xs = np.linspace(x0, x1, 2 * nx + 1)
    ys = np.linspace(0.0, height, 2 * ny + 1)
    xx, yy = np.meshgrid(xs, ys)
    return np.column_stack((xx.ravel(), yy.ravel()))


def _node_sets(nodes: NDArray[np.float64], cut: float, *, clamp: bool):
    """Split nodes into retained, condensed, and clamped sets.

    Parameters
    ----------
    nodes
        Node coordinates, one row per node.
    cut
        Abscissa of the interface. Nodes on this line are retained.
    clamp
        If true, nodes at :math:`x = 0` are clamped. Otherwise the clamped
        set is empty.

    Returns
    -------
    retained : ndarray
        Nodes on the cut.
    condensed : ndarray
        Free nodes that are not on the cut.
    clamped : ndarray
        Nodes on the wall. Empty when ``clamp`` is false.
    """
    on_cut = np.isclose(nodes[:, 0], cut)
    on_wall = np.isclose(nodes[:, 0], 0.0) if clamp else np.zeros(nodes.shape[0], dtype=bool)
    return nodes[on_cut], nodes[~(on_cut | on_wall)], nodes[on_wall]


def _draw_condensation_panel(
    axis,
    left_mesh,
    right_mesh,
    left_sets,
    right_sets,
    shift,
    wall,
    xlim,
    ylim,
    show_condensed,
) -> None:
    """Draw one row of the static-condensation figure.

    Parameters
    ----------
    axis : Axes
        Axes to draw on.
    left_mesh : Mesh
        Mesh of the clamped piece.
    right_mesh : Mesh
        Mesh of the free piece.
    left_sets : tuple of ndarray
        Retained, condensed, and clamped nodes of the clamped piece.
    right_sets : tuple of ndarray
        Retained, condensed, and clamped nodes of the free piece.
    shift : float
        Half-gap by which the pieces are pulled apart.
    wall : float
        Abscissa of the drawn support.
    xlim : tuple of float
        Horizontal limits.
    ylim : tuple of float
        Vertical limits.
    show_condensed : bool
        If true, condensed nodes are drawn as open circles.
    """
    _draw_colored_elements(axis, left_mesh, np.ones(left_mesh.t.shape[1], dtype=bool), "#c6dbef", -shift)
    _draw_colored_elements(axis, right_mesh, np.ones(right_mesh.t.shape[1], dtype=bool), "#fdd0a2", shift)
    y0 = float(min(left_mesh.p[1].min(), right_mesh.p[1].min()))
    y1 = float(max(left_mesh.p[1].max(), right_mesh.p[1].max()))
    _draw_wall(axis, wall, y0, y1)
    _scatter_node_sets(axis, left_sets, -shift, show_condensed)
    _scatter_node_sets(axis, right_sets, shift, show_condensed)
    _style_mesh_axis(axis, xlim, ylim)


def _scatter_node_sets(axis, sets, shift: float, show_condensed: bool) -> None:
    """Draw retained, condensed, and clamped nodes.

    Parameters
    ----------
    axis : Axes
        Axes to draw on.
    sets : tuple of ndarray
        Retained nodes, condensed nodes, and clamped nodes.
    shift
        Horizontal offset added to every node.
    show_condensed
        If true, condensed nodes are drawn as open circles.
    """
    retained, condensed, clamped = sets
    if show_condensed and condensed.size:
        axis.scatter(
            condensed[:, 0] + shift,
            condensed[:, 1],
            s=28,
            facecolors="white",
            edgecolors="0.15",
            linewidths=0.9,
            zorder=5,
        )
    if clamped.size:
        axis.scatter(clamped[:, 0] + shift, clamped[:, 1], s=22, c="0.45", marker="s", zorder=5)
    if retained.size:
        axis.scatter(retained[:, 0] + shift, retained[:, 1], s=28, c="0.1", zorder=6)


def _draw_assembly_panel(axis, mesh, left, right, interface, shift, wall, xlim, ylim) -> None:
    """Draw the two pieces of one assembly panel, apart or joined.

    Parameters
    ----------
    axis : Axes
        Axes to draw on.
    mesh : Mesh
        Quadrilateral mesh of the unreduced beam.
    left : ndarray
        Boolean mask of the elements on the clamped side of the cut.
    right : ndarray
        Boolean mask of the elements on the free side of the cut.
    interface : ndarray
        Coordinates of the interface nodes, shape ``(2, n_nodes)``.
    shift : float
        Half-gap by which the pieces are pulled apart. Zero draws them joined.
    wall : float
        Abscissa of the drawn support.
    xlim : tuple of float
        Horizontal limits.
    ylim : tuple of float
        Vertical limits.
    """
    _draw_colored_elements(axis, mesh, left, "#c6dbef", -shift)
    _draw_colored_elements(axis, mesh, right, "#fdd0a2", shift)
    _draw_wall(axis, wall, float(interface[1].min()), float(interface[1].max()))
    nodes_x = interface[0] - shift
    axis.scatter(nodes_x, interface[1], s=16, c="0.1", zorder=5)
    if shift:
        axis.scatter(interface[0] + shift, interface[1], s=16, c="0.1", zorder=5)
    _style_mesh_axis(axis, xlim, ylim)


def _style_mesh_axis(axis, xlim, ylim) -> None:
    """Set equal limits and hide the frame.

    Parameters
    ----------
    axis : Axes
        Axes to style.
    xlim : tuple of float
        Horizontal limits.
    ylim : tuple of float
        Vertical limits.
    """
    axis.set_xlim(*xlim)
    axis.set_ylim(*ylim)
    axis.set_aspect("equal", adjustable="box")
    axis.set_xticks([])
    axis.set_yticks([])
    for spine in axis.spines.values():
        spine.set_visible(False)


def _draw_colored_elements(axis, mesh, elements, facecolor, shift: float) -> None:
    """Draw a subset of elements, shifted along :math:`x`.

    Parameters
    ----------
    axis : Axes
        Axes to draw on.
    mesh : Mesh
        Quadrilateral mesh.
    elements : ndarray
        Boolean mask of the elements to draw.
    facecolor : str
        Face colour of those elements.
    shift
        Horizontal offset added to the element coordinates.
    """
    polygons = np.moveaxis(mesh.p.T[mesh.t[:, elements]], 0, 1).astype(float, copy=True)
    polygons[..., 0] += shift
    axis.add_collection(
        PolyCollection(polygons, facecolors=facecolor, edgecolors="0.35", linewidths=0.35)
    )


def _draw_wall(axis, x: float, y0: float, y1: float) -> None:
    """Draw the clamped support.

    Parameters
    ----------
    axis : Axes
        Axes to draw on.
    x
        Abscissa of the wall.
    y0
        Lower end of the wall.
    y1
        Upper end of the wall.
    """
    axis.plot([x, x], [y0, y1], color="0.1", linewidth=2, zorder=4, solid_capstyle="butt")
    for tick in np.linspace(y0, y1, 8):
        axis.plot([x - 0.05, x], [tick - 0.025, tick], color="0.1", linewidth=0.8, zorder=4)


def _draw_interface_forces(axis, interface, gap: float) -> None:
    """Draw a pair of opposing arrows at each interface node.

    Parameters
    ----------
    axis : Axes
        Axes to draw on.
    interface : ndarray
        Coordinates of the interface nodes, shape ``(2, n_nodes)``.
    gap
        Distance between the two pieces. The arrows point across that gap.
    """
    length = 0.42 * gap
    zeros = np.zeros(interface.shape[1])
    arrows = np.full(interface.shape[1], length)
    shared = dict(angles="xy", scale_units="xy", scale=1, width=0.003, color="0.1", zorder=6)
    axis.quiver(interface[0] - gap / 2, interface[1], arrows, zeros, **shared)
    axis.quiver(interface[0] + gap / 2, interface[1], -arrows, zeros, **shared)


def _mark_interface(axis, x, y, symbol: str) -> None:
    """Label the uppermost interface node.

    Parameters
    ----------
    axis : Axes
        Axes to annotate.
    x : ndarray
        Horizontal coordinates of the interface nodes.
    y : ndarray
        Vertical coordinates of the interface nodes.
    symbol
        Annotation text, such as the displacement or force unknown.
    """
    top = int(np.argmax(y))
    axis.annotate(
        symbol,
        (x[top], y[top]),
        textcoords="offset points",
        xytext=(0, 8),
        ha="center",
        va="bottom",
        fontsize=13,
    )


def _slug(label: str) -> str:
    """Turn a mode title into a file-name fragment.

    Parameters
    ----------
    label
        Title text. Characters that are not letters or digits become underscores.

    Returns
    -------
    str
        Lower-case fragment with no leading, trailing, or repeated underscores.
    """
    characters = [character.lower() if character.isalnum() else "_" for character in label]
    slug = "".join(characters).strip("_")
    while "__" in slug:
        slug = slug.replace("__", "_")
    return slug


def _vertex_motion(model: Model, displacement: NDArray[np.float64]) -> NDArray[np.float64]:
    """Gather the displacement of each mesh vertex.

    Parameters
    ----------
    model
        Component that owns ``displacement``. A clamped vertex unknown is
        stored as ``-1`` and contributes no motion.
    displacement
        Displacement of the free degrees of freedom.

    Returns
    -------
    ndarray
        Vertex motion, shape ``(n_vertices, 2)``.
    """
    motion = np.zeros((model.vertex_dofs.shape[0], model.vertex_dofs.shape[1]))
    for component in range(model.vertex_dofs.shape[1]):
        index = model.vertex_dofs[:, component]
        active = index >= 0
        motion[active, component] = displacement[index[active]]
    return motion


def _beam_limits(meshes, deflection: float):
    """Axis limits that leave room for the drawn deflection.

    Parameters
    ----------
    meshes : sequence of Mesh
        Meshes whose bounding box sets the view.
    deflection
        Padding on every side, as a fraction of the beam height.

    Returns
    -------
    xlim : tuple of float
        Horizontal limits.
    ylim : tuple of float
        Vertical limits.
    """
    points = np.hstack([mesh.p for mesh in meshes])
    x0, x1 = float(points[0].min()), float(points[0].max())
    y0, y1 = float(points[1].min()), float(points[1].max())
    pad = deflection * (y1 - y0)
    return (x0 - pad, x1 + pad), (y0 - pad, y1 + pad)


def _deformed_positions(mesh, motion, meshes, deflection: float):
    """Scale a vertex motion so its peak is a fraction of the beam height.

    Parameters
    ----------
    mesh : Mesh
        Mesh whose vertex coordinates are shifted.
    motion : ndarray
        Vertex motion, shape ``(n_vertices, 2)``.
    meshes : sequence of Mesh
        Meshes used to measure the beam height.
    deflection
        Peak drawn amplitude as a fraction of that height. A zero motion is
        returned unscaled.

    Returns
    -------
    ndarray
        Deformed vertex coordinates, shape ``(n_vertices, 2)``.
    """
    peak = np.max(np.linalg.norm(motion, axis=1))
    if peak == 0.0:
        return mesh.p.T
    height = max(float(np.ptp(item.p[1])) for item in meshes)
    return mesh.p.T + (deflection * height / peak) * motion


def _draw_ghost(axis, mesh) -> None:
    """Draw an undeformed mesh in gray.

    Parameters
    ----------
    axis : Axes
        Axes to draw on.
    mesh : Mesh
        Mesh drawn in place, without deformation.
    """
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
    """Draw a deformed mesh coloured by the displacement magnitude.

    Parameters
    ----------
    axis : Axes
        Axes to draw on.
    mesh : Mesh
        Element connectivity. The vertex positions come from ``positions``.
    positions : ndarray
        Deformed vertex coordinates, shape ``(n_vertices, 2)``.
    motion : ndarray
        Vertex motion used for the colour, shape ``(n_vertices, 2)``. Each
        element is coloured by its mean magnitude, divided by the peak.

    Returns
    -------
    PolyCollection
        Collection added to ``axis``.
    """
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
