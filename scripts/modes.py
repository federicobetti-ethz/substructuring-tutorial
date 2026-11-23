"""Draw the Craig–Bampton and Rubin shapes of the cantilever components.

Run from the repository root:

    poetry run python scripts/modes.py
"""

from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from substructuring import (  # noqa: E402
    Material,
    craig_bampton,
    mid_interface_column,
    plot_modes,
    rectangle_mesh,
    rigid_body_modes,
    rubin,
    structural_model,
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
    craig_left = craig_bampton(left.stiffness, left.mass, left.boundary, N_MODES)
    craig_right = craig_bampton(right.stiffness, right.mass, right.boundary, N_MODES)
    rubin_left = rubin(left.stiffness, left.mass, left.boundary, N_MODES, n_rigid=0)
    rubin_right = rubin(right.stiffness, right.mass, right.boundary, N_MODES, n_rigid=3)

    craig_path = plot_modes(
        (
            ("Left substructure", left_mesh, left, _craig_bampton_entries(left, craig_left)),
            ("Right substructure", right_mesh, right, _craig_bampton_entries(right, craig_right)),
        ),
        FIGURES / "craig_bampton_modes.png",
        suptitle="Craig–Bampton modes of each substructure",
    )
    rubin_path = plot_modes(
        (
            ("Left substructure", left_mesh, left, _rubin_clamped_entries(left, rubin_left)),
            ("Right substructure", right_mesh, right, _rubin_free_entries(right, rubin_right)),
        ),
        FIGURES / "rubin_modes.png",
        suptitle="Rubin modes of each substructure",
    )
    print(f"Wrote {craig_path}")
    print(f"Wrote {rubin_path}")


def cantilever():
    half = NX // 2
    left_mesh = rectangle_mesh(0.0, LENGTH / 2.0, HEIGHT, half, NY)
    right_mesh = rectangle_mesh(LENGTH / 2.0, LENGTH, HEIGHT, half, NY)
    left = structural_model(left_mesh, MATERIAL, interface="right", clamp="left")
    right = structural_model(right_mesh, MATERIAL, interface="left")
    return left_mesh, right_mesh, left, right


def _craig_bampton_entries(model, reduced):
    constraint = _family(reduced, "constraint")
    vibrating = _family(reduced, "fixed interface")
    column = mid_interface_column(model, component=1)
    entries = [("Constraint, vertical", constraint.values[:, column])]
    entries.extend(_vibrating(vibrating, N_MODES))
    return entries


def _rubin_clamped_entries(model, reduced):
    attachment = _family(reduced, "attachment")
    vibrating = _family(reduced, "free interface")
    column = mid_interface_column(model, component=1)
    entries = [("Attachment, vertical", attachment.values[:, column])]
    entries.extend(_vibrating(vibrating, N_MODES))
    return [None, None, None] + entries


def _rubin_free_entries(model, reduced):
    labels = ("Rigid translation x", "Rigid translation y", "Rigid rotation")
    rigid = [
        (label, shape)
        for label, shape in zip(labels, rigid_body_modes(model).T)
    ]
    attachment = _family(reduced, "attachment")
    column = mid_interface_column(model, component=1)
    elastic = _vibrating(_family(reduced, "free interface"), N_MODES)
    return rigid + [("Attachment, vertical", attachment.values[:, column])] + elastic


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
