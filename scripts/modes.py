"""Draw the Craig–Bampton and Rubin shapes of the cantilever components.

Run from the repository root:

    poetry run python scripts/modes.py
"""

import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from substructuring import (  # noqa: E402
    Material,
    chain,
    plot_assembly,
    plot_model,
    plot_modes,
    plot_static_condensation,
    plot_vibration_modes,
    rectangle_mesh,
    rigid_body_modes,
    structural_model,
    vibration_modes,
)

LENGTH = 2.0
HEIGHT = 0.35
NX = 40
NY = 10
N_MODES = 3
MATERIAL = Material(young=210.0e9, poisson=0.3, density=7800.0, thickness=0.01)
FIGURES = ROOT / "figures"


def main() -> None:
    left_mesh, right_mesh, left, right = cantilever()
    pieces = chain([left, right])
    craig_left, craig_right = pieces.reduce(
        "craig-bampton", n_modes=N_MODES
    ).substructures
    rubin_left, rubin_right = pieces.reduce(
        "rubin", n_modes=N_MODES, n_rigid=[0, 3]
    ).substructures

    craig_dir = FIGURES / "craig_bampton"
    for stale in craig_dir.glob("*.png"):
        stale.unlink()
    craig_paths = plot_modes(
        (
            ("Left", left_mesh, left, _craig_bampton_entries(left, craig_left)),
            ("Right", right_mesh, right, _craig_bampton_entries(right, craig_right)),
        ),
        FIGURES / "craig_bampton",
    )
    rubin_dir = FIGURES / "rubin"
    for stale in rubin_dir.glob("*.png"):
        stale.unlink()
    rubin_paths = plot_modes(
        (
            ("Left", left_mesh, left, _rubin_clamped_entries(left, rubin_left)),
            ("Right", right_mesh, right, _rubin_free_entries(right, rubin_right)),
        ),
        FIGURES / "rubin",
    )
    mesh = rectangle_mesh(0.0, LENGTH, HEIGHT, NX, NY)
    full = structural_model(mesh, MATERIAL, clamp="left")
    vibration_paths = plot_vibration_modes(
        mesh,
        full,
        vibration_modes(full, N_MODES),
        FIGURES / "vibration",
    )
    model_path = plot_model(mesh, FIGURES / "model.png")
    primal, dual = plot_assembly(
        mesh, FIGURES / "primal.png", FIGURES / "dual.png", cut=0.5 * LENGTH
    )
    condensation = plot_static_condensation(
        FIGURES / "condensation.png", length=LENGTH, height=HEIGHT
    )
    for path in (
        *craig_paths,
        *rubin_paths,
        *vibration_paths,
        model_path,
        primal,
        dual,
        condensation,
    ):
        print(f"Wrote {path}")


def cantilever():
    half = NX // 2
    left_mesh = rectangle_mesh(0.0, LENGTH / 2.0, HEIGHT, half, NY)
    right_mesh = rectangle_mesh(LENGTH / 2.0, LENGTH, HEIGHT, half, NY)
    left = structural_model(left_mesh, MATERIAL, interface="right", clamp="left")
    right = structural_model(right_mesh, MATERIAL, interface="left")
    return left_mesh, right_mesh, left, right


def _craig_bampton_entries(model, reduced):
    """Static response of the piece to a rigid motion of the whole cut."""
    constraint = _family(reduced, "constraint")
    labels = (
        "Constraint translation x",
        "Constraint translation y",
        "Constraint rotation",
    )
    motions = _interface_motions(model)
    entries = [
        (label, constraint.values @ motions[:, index])
        for index, label in enumerate(labels)
    ]
    entries.extend(_vibrating(_family(reduced, "fixed interface"), N_MODES))
    return entries


def _interface_motions(model):
    """Horizontal, vertical, and rotational displacement of the interface.

    Columns follow ``model.boundary``. The rotation spins the cut about its
    midpoint and is scaled so the largest interface displacement is 1. On this
    vertical cut that spin is a horizontal displacement linear in height.
    """
    coordinates = model.coordinates[model.boundary]
    component = model.component[model.boundary]
    motion = np.zeros((model.boundary.size, 3))
    motion[component == 0, 0] = 1.0
    motion[component == 1, 1] = 1.0
    center = 0.5 * (coordinates[:, 1].min() + coordinates[:, 1].max())
    spin = np.zeros(model.boundary.size)
    spin[component == 0] = -(coordinates[component == 0, 1] - center)
    peak = np.max(np.abs(spin))
    if peak > 0.0:
        spin /= peak
    motion[:, 2] = spin
    return motion


def _rubin_clamped_entries(model, reduced):
    vibrating = _vibrating(_family(reduced, "free interface"), N_MODES)
    return [None, None, None] + _residual_entries(model, reduced) + vibrating


def _rubin_free_entries(model, reduced):
    labels = ("Rigid translation x", "Rigid translation y", "Rigid rotation")
    rigid = [(label, shape) for label, shape in zip(labels, rigid_body_modes(model).T)]
    elastic = _vibrating(_family(reduced, "free interface"), N_MODES)
    return rigid + _residual_entries(model, reduced) + elastic


def _residual_entries(model, reduced):
    """Residual flexibility for a rigid motion of the whole cut."""
    attachment = _family(reduced, "attachment")
    labels = ("Residual translation x", "Residual translation y", "Residual rotation")
    motions = _interface_motions(model)
    return [
        (label, attachment.values @ motions[:, index])
        for index, label in enumerate(labels)
    ]


def _family(reduced, label):
    return next(shape for shape in reduced.shapes if shape.label == label)


def _vibrating(shape, count):
    shown = min(count, shape.values.shape[1])
    return [
        (f"{shape.frequencies[index]:.0f} Hz", shape.values[:, index])
        for index in range(shown)
    ]


if __name__ == "__main__":
    main()
