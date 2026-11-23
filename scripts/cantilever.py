"""Craig–Bampton and Rubin reduction of a plane-stress cantilever.

The beam is a rectangle of length L and height L/10, clamped at x = 0 and cut
at midspan. Each half is a quadratic quadrilateral mesh. Both reductions are
repeated with more component modes, and the error is the relative L2 norm of
the complex tip receptance against the full finite-element model.

Run from the repository root:

    poetry run python scripts/cantilever.py
"""

from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
from numpy.typing import NDArray
from scipy.linalg import eigh

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from substructuring import (  # noqa: E402
    Material,
    couple,
    craig_bampton,
    rectangle_mesh,
    rubin,
    structural_model,
)
from substructuring.coupling import CoupledModel, ReducedModel  # noqa: E402


LENGTH = 1.0
HEIGHT = 0.1
NX = 16
NY = 4
MODE_COUNTS = (2, 8, 16, 32)
N_FREQUENCIES = 8
N_SAMPLES = 800
STRUCTURAL_DAMPING = 0.01
MATERIAL = Material(young=210.0e9, poisson=0.3, density=7800.0, thickness=0.01)


def main() -> None:
    check_craig_bampton_is_complete()
    left, right, full = cantilever(NX, NY)
    print(
        f"Free degrees of freedom: full {full.stiffness.shape[0]}, "
        f"left {left.stiffness.shape[0]}, right {right.stiffness.shape[0]}, "
        f"interface {left.boundary.size}"
    )

    samples, reference = reference_receptance(full)
    craig_errors = []
    rubin_errors = []
    receptances = {}
    for n_modes in MODE_COUNTS:
        craig, craig_right, rubin_model, rubin_right = reductions(left, right, n_modes)
        craig_response = component_receptance(right, craig_right, craig, samples)
        rubin_response = component_receptance(right, rubin_right, rubin_model, samples)
        craig_errors.append(relative_l2(reference, craig_response))
        rubin_errors.append(relative_l2(reference, rubin_response))
        receptances[n_modes] = (craig_response, rubin_response)

    report_transfer_errors(samples, craig_errors, rubin_errors)
    figure = transfer_error_figure(samples, reference, receptances)
    print(f"Wrote {figure}")


def cantilever(nx: int, ny: int):
    """Clamp the left end and match both sides of the midspan cut."""
    if nx % 2:
        raise ValueError("Use an even number of elements along the beam")
    half = nx // 2
    left = structural_model(
        rectangle_mesh(0.0, LENGTH / 2.0, HEIGHT, half, ny),
        MATERIAL,
        interface="right",
        clamp="left",
    )
    right = structural_model(
        rectangle_mesh(LENGTH / 2.0, LENGTH, HEIGHT, half, ny),
        MATERIAL,
        interface="left",
    )
    full = structural_model(
        rectangle_mesh(0.0, LENGTH, HEIGHT, nx, ny),
        MATERIAL,
        clamp="left",
    )
    same_interface = np.allclose(
        left.coordinates[left.boundary],
        right.coordinates[right.boundary],
        atol=1e-10,
    )
    if not same_interface:
        raise RuntimeError("The two interface meshes do not carry the same degrees of freedom")
    return left, right, full


def check_craig_bampton_is_complete() -> None:
    """Every fixed-interface mode makes the coupled Craig–Bampton model exact."""
    left, right, full = cantilever(4, 2)
    coupled = couple(
        (
            craig_bampton(
                left.stiffness,
                left.mass,
                left.boundary,
                left.stiffness.shape[0] - left.boundary.size,
            ),
            craig_bampton(
                right.stiffness,
                right.mass,
                right.boundary,
                right.stiffness.shape[0] - right.boundary.size,
            ),
        )
    )
    reference = frequencies(full.stiffness, full.mass, 12)
    obtained = frequencies(coupled.stiffness, coupled.mass, 12)
    if not np.allclose(obtained, reference, rtol=1e-6, atol=1e-6):
        raise RuntimeError("Complete Craig–Bampton reduction does not recover the full model")


def reductions(left, right, n_modes: int):
    """Craig–Bampton and Rubin models with the same number of elastic modes."""
    craig_right = craig_bampton(right.stiffness, right.mass, right.boundary, n_modes)
    rubin_right = rubin(right.stiffness, right.mass, right.boundary, n_modes, n_rigid=3)
    craig = couple(
        (
            craig_bampton(left.stiffness, left.mass, left.boundary, n_modes),
            craig_right,
        )
    )
    rubin_model = couple(
        (
            rubin(left.stiffness, left.mass, left.boundary, n_modes, n_rigid=0),
            rubin_right,
        )
    )
    return craig, craig_right, rubin_model, rubin_right


def reference_receptance(full):
    """Tip receptance of the unreduced cantilever over the first elastic modes."""
    band = frequencies(full.stiffness, full.mass, N_FREQUENCIES)
    samples = np.geomspace(0.5 * band[0], 1.2 * band[-1], N_SAMPLES)
    sensor = tip_dof(full)
    force = np.zeros(full.stiffness.shape[0])
    force[sensor] = 1.0
    response = modal_receptance(full.stiffness, full.mass, force, sensor, samples)
    return samples, response


def component_receptance(right, reduced: ReducedModel, coupled: CoupledModel, samples):
    """Tip receptance recovered from one coupled reduction."""
    sensor = tip_dof(right)
    force = np.zeros(right.stiffness.shape[0])
    force[sensor] = 1.0
    return modal_receptance(
        coupled.stiffness,
        coupled.mass,
        reduced_force(reduced, force, coupled, 1),
        None,
        samples,
        sensor_row=recovery_row(reduced, sensor, coupled, 1),
    )


def report_transfer_errors(samples, craig_errors, rubin_errors) -> None:
    print(
        "Relative L2 error of the complex tip receptance, "
        f"{samples[0]:.1f}-{samples[-1]:.1f} Hz"
    )
    header = f"{'modes':>6}  {'Craig-Bampton':>14}  {'Rubin':>14}"
    print(header)
    print("-" * len(header))
    for n_modes, craig_error, rubin_error in zip(MODE_COUNTS, craig_errors, rubin_errors):
        print(f"{n_modes:6d}  {craig_error:14.3e}  {rubin_error:14.3e}")
    print(
        "The error is ||H_reduced - H_full|| / ||H_full|| on the frequency samples. "
        "Each count is the number of elastic modes kept in each component."
    )


def transfer_error_figure(samples, reference, receptances):
    """Pointwise relative error of the tip receptance against frequency."""
    figure, axis = plt.subplots(figsize=(7.2, 4.4))
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    for color, n_modes in zip(colors, MODE_COUNTS):
        craig_response, rubin_response = receptances[n_modes]
        axis.loglog(samples, _relative_error(rubin_response, reference), color=color, linestyle="-")
        axis.loglog(samples, _relative_error(craig_response, reference), color=color, linestyle="--")
    axis.set_xlabel("Frequency (Hz)")
    axis.set_ylabel("Relative error")
    axis.grid(True, which="both", alpha=0.3)
    axis.legend(handles=_error_legend(colors), fontsize="small")
    figure.tight_layout()
    output = ROOT / "figures" / "transfer_error.png"
    output.parent.mkdir(exist_ok=True)
    figure.savefig(output, dpi=150)
    plt.close(figure)
    return output


def _relative_error(response, reference):
    return np.abs(response - reference) / np.abs(reference)


def _error_legend(colors):
    style = [
        Line2D([0], [0], color="black", linestyle="-", label="Rubin"),
        Line2D([0], [0], color="black", linestyle="--", label="Craig–Bampton"),
    ]
    counts = [
        Line2D([0], [0], color=color, linestyle="-", label=f"{n_modes} modes")
        for color, n_modes in zip(colors, MODE_COUNTS)
    ]
    return style + counts


def reduced_force(component: ReducedModel, force, coupled: CoupledModel, component_index: int):
    """Project a force on one component into the coupled reduced coordinates."""
    local_force = component.basis.T @ force
    assembled = np.zeros(coupled.stiffness.shape[0])
    assembled[: coupled.n_boundary] = local_force[: component.n_boundary]
    assembled[coupled.local[component_index]] = local_force[component.n_boundary :]
    return assembled


def recovery_row(component: ReducedModel, dof: int, coupled: CoupledModel, component_index: int):
    """Row that recovers one finite-element unknown from the coupled coordinates."""
    row = np.zeros(coupled.stiffness.shape[0])
    row[: coupled.n_boundary] = component.basis[dof, : component.n_boundary]
    row[coupled.local[component_index]] = component.basis[dof, component.n_boundary :]
    return row


def modal_receptance(stiffness, mass, force, sensor, samples, sensor_row=None):
    """Receptance with stiffness-proportional structural damping."""
    eigenvalues, modes = eigh(stiffness, mass)
    eigenvalues = np.maximum(eigenvalues, 0.0)
    modal_force = modes.T @ force
    if sensor_row is None:
        modal_sensor = modes[sensor]
    else:
        modal_sensor = sensor_row @ modes
    frequency = (2.0 * np.pi * samples) ** 2
    denominator = eigenvalues[:, None] * (1.0 + 1j * STRUCTURAL_DAMPING) - frequency[None, :]
    modal_coordinates = modal_force[:, None] / denominator
    return modal_sensor @ modal_coordinates


def tip_dof(model) -> int:
    """Vertical unknown closest to the middle of the free end."""
    vertical = np.flatnonzero(model.component == 1)
    coordinates = model.coordinates[vertical]
    distance = (coordinates[:, 0] - LENGTH) ** 2 + (coordinates[:, 1] - 0.5 * HEIGHT) ** 2
    return int(vertical[np.argmin(distance)])


def frequencies(stiffness, mass, count: int) -> NDArray[np.float64]:
    values = eigh(stiffness, mass, eigvals_only=True, subset_by_index=[0, count - 1])
    return np.sqrt(np.maximum(values, 0.0)) / (2.0 * np.pi)


def relative_l2(reference, approximation) -> float:
    return float(np.linalg.norm(approximation - reference) / np.linalg.norm(reference))


if __name__ == "__main__":
    main()
