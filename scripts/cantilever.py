"""Craig–Bampton and Rubin reduction of a plane-stress cantilever."""

from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
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


LENGTH = 2.0
HEIGHT = 0.35
NX = 40
NY = 10
N_SUBSTRUCTURES = 2
MODE_COUNTS = (2, 8, 16, 32)
DAMPING_RATIO = 0.01
CYCLES = 6.0
RAMP_CYCLES = 0.08
TIP_FORCE = 10_000.0
N_TIMES = 2000
MATERIAL = Material(young=210.0e9, poisson=0.3, density=7800.0, thickness=0.01)


def main() -> None:
    check_craig_bampton_is_complete()
    full = structural_model(
        rectangle_mesh(0.0, LENGTH, HEIGHT, NX, NY),
        MATERIAL,
        clamp="left",
    )
    fundamental = frequencies(full.stiffness, full.mass, 1)[0]
    ramp = RAMP_CYCLES / fundamental
    times = np.linspace(0.0, CYCLES / fundamental, N_TIMES)
    print(f"Free degrees of freedom in the full model: {full.stiffness.shape[0]}")
    print(
        f"{TIP_FORCE:.0f} N in x and y, raised over {1e3 * ramp:.2f} ms and then held, "
        f"followed until {1e3 * times[-1]:.1f} ms"
    )
    reference = tip_history(full.stiffness, full.mass, _tip_force(full), _tip_rows(full), times, ramp)
    components = beam_components(N_SUBSTRUCTURES, NX, NY)
    tip = components[-1]
    print(f"{N_SUBSTRUCTURES} substructures, {tip.boundary.size} degrees of freedom per cut")
    results = {}
    for n_modes in MODE_COUNTS:
        craig, rubin_model, craig_tip, rubin_tip = reduce_chain(components, n_modes)
        print(
            f"{n_modes} elastic modes, reduced size {craig.stiffness.shape[0]} (Craig-Bampton) "
            f"and {rubin_model.stiffness.shape[0]} (Rubin)"
        )
        results[n_modes] = (
            component_history(tip, craig_tip, craig, times, ramp),
            component_history(tip, rubin_tip, rubin_model, times, ramp),
            craig.stiffness.shape[0],
            rubin_model.stiffness.shape[0],
        )
    report_tip_errors(reference, results)
    history = tip_history_figure(times, reference, results)
    error = tip_norm_error_figure(reference, results)
    print(f"Wrote {history}")
    print(f"Wrote {error}")


def beam_components(n_sub: int, nx: int, ny: int):
    """Clamp the root and cut the beam into ``n_sub`` equal pieces."""
    if n_sub < 2:
        raise ValueError("A chain needs at least two substructures")
    if nx % n_sub:
        raise ValueError(f"{nx} elements along the beam do not divide into {n_sub} pieces")
    nx_piece = nx // n_sub
    stations = np.linspace(0.0, LENGTH, n_sub + 1)
    components = []
    for index in range(n_sub):
        mesh = rectangle_mesh(stations[index], stations[index + 1], HEIGHT, nx_piece, ny)
        if index == 0:
            interface = "right"
            clamp = "left"
        elif index == n_sub - 1:
            interface = "left"
            clamp = None
        else:
            interface = ("left", "right")
            clamp = None
        components.append(structural_model(mesh, MATERIAL, interface=interface, clamp=clamp))
    _check_cuts_match(components)
    return components


def _check_cuts_match(components) -> None:
    right = components[0].boundary
    for index in range(1, len(components)):
        left = components[index].boundary[: right.size]
        same = np.allclose(
            components[index - 1].coordinates[right],
            components[index].coordinates[left],
            atol=1e-10,
        )
        if not same or left.size != right.size:
            raise RuntimeError(f"Cut {index - 1} does not have matching degrees of freedom")
        if index < len(components) - 1:
            right = components[index].boundary[right.size :]


def check_craig_bampton_is_complete() -> None:
    """Every fixed-interface mode makes a chain of Craig–Bampton models exact."""
    for n_sub, nx in ((2, 4), (3, 6)):
        components = beam_components(n_sub, nx, 2)
        full = structural_model(
            rectangle_mesh(0.0, LENGTH, HEIGHT, nx, 2),
            MATERIAL,
            clamp="left",
        )
        coupled = couple(
            [
                craig_bampton(
                    component.stiffness,
                    component.mass,
                    component.boundary,
                    component.stiffness.shape[0] - component.boundary.size,
                )
                for component in components
            ]
        )
        reference = frequencies(full.stiffness, full.mass, 12)
        obtained = frequencies(coupled.stiffness, coupled.mass, 12)
        if not np.allclose(obtained, reference, rtol=1e-6, atol=1e-6):
            raise RuntimeError(
                f"Complete Craig–Bampton reduction with {n_sub} substructures "
                "does not recover the full model"
            )


def reduce_chain(components, n_modes: int):
    """Craig–Bampton and Rubin models with the same number of elastic modes."""
    craig_parts = []
    rubin_parts = []
    for index, component in enumerate(components):
        craig_parts.append(
            craig_bampton(component.stiffness, component.mass, component.boundary, n_modes)
        )
        rubin_parts.append(
            rubin(
                component.stiffness,
                component.mass,
                component.boundary,
                n_modes,
                n_rigid=0 if index == 0 else 3,
            )
        )
    return couple(craig_parts), couple(rubin_parts), craig_parts[-1], rubin_parts[-1]


def tip_history(stiffness, mass, force, rows, times, ramp) -> NDArray[np.float64]:
    """Displacement at the requested rows under the shared tip load."""
    eigenvalues, modes = eigh(stiffness, mass)
    omega = np.sqrt(np.maximum(eigenvalues, 0.0))
    if np.any(omega <= 1e-8 * np.max(omega)):
        raise RuntimeError("The coupled model still has a rigid-body mode")
    modal_force = modes.T @ force
    modal = _ramp_response(omega, modal_force, times, ramp)
    return (rows @ modes) @ modal


def component_history(tip, reduced: ReducedModel, coupled: CoupledModel, times, ramp):
    """Tip displacement recovered from the last component of a coupled chain."""
    component_index = len(coupled.local) - 1
    horizontal, vertical = tip_dofs(tip)
    force = reduced_force(reduced, _tip_load(tip, horizontal, vertical), coupled, component_index)
    rows = np.vstack(
        (
            recovery_row(reduced, horizontal, coupled, component_index),
            recovery_row(reduced, vertical, coupled, component_index),
        )
    )
    return tip_history(coupled.stiffness, coupled.mass, force, rows, times, ramp)


def _tip_force(model) -> NDArray[np.float64]:
    horizontal, vertical = tip_dofs(model)
    return _tip_load(model, horizontal, vertical)


def _tip_load(model, horizontal: int, vertical: int) -> NDArray[np.float64]:
    force = np.zeros(model.stiffness.shape[0])
    force[horizontal] = TIP_FORCE
    force[vertical] = TIP_FORCE
    return force


def _tip_rows(model) -> NDArray[np.float64]:
    horizontal, vertical = tip_dofs(model)
    rows = np.zeros((2, model.stiffness.shape[0]))
    rows[0, horizontal] = 1.0
    rows[1, vertical] = 1.0
    return rows


def _ramp_response(omega, modal_force, times, ramp) -> NDArray[np.float64]:
    """Exact modal response to a raised-cosine ramp that then holds.

    The load factor is ``(1 - cos(π t / T)) / 2`` until ``T`` and 1 afterwards.
    Each oscillator is ``s'' + 2 ζ ω s' + ω² s = f p(t)``, at rest initially.
    ``ζ`` is :data:`DAMPING_RATIO`.
    """
    zeta = DAMPING_RATIO
    drive = np.pi / ramp
    natural = omega**2
    det = (natural - drive**2) ** 2 + (2.0 * zeta * omega * drive) ** 2
    half = -0.5 * modal_force
    cosine = half * (natural - drive**2) / det
    sine = half * (2.0 * zeta * omega * drive) / det
    static_half = 0.5 * modal_force / natural
    decay_rate = zeta * omega
    damped = omega * np.sqrt(1.0 - zeta**2)
    cosine_state = -static_half - cosine
    sine_state = (-sine * drive + decay_rate * cosine_state) / damped

    rising = _rising_response(
        times, drive, decay_rate, damped, static_half, cosine, sine, cosine_state, sine_state
    )
    end = _rising_state(ramp, drive, decay_rate, damped, static_half, cosine, sine, cosine_state, sine_state)
    held = modal_force / natural
    held_cosine = end[0] - held
    held_sine = (end[1] + decay_rate * held_cosine) / damped
    elapsed = times - ramp
    settled = held[:, None] + np.exp(-decay_rate[:, None] * np.maximum(elapsed, 0.0)[None, :]) * (
        held_cosine[:, None] * np.cos(damped[:, None] * elapsed[None, :])
        + held_sine[:, None] * np.sin(damped[:, None] * elapsed[None, :])
    )
    return np.where(times[None, :] <= ramp, rising, settled)


def _rising_response(times, drive, decay_rate, damped, static_half, cosine, sine, cosine_state, sine_state):
    decay = np.exp(-decay_rate[:, None] * times[None, :])
    homogeneous = decay * (
        cosine_state[:, None] * np.cos(damped[:, None] * times[None, :])
        + sine_state[:, None] * np.sin(damped[:, None] * times[None, :])
    )
    particular = (
        static_half[:, None]
        + cosine[:, None] * np.cos(drive * times)[None, :]
        + sine[:, None] * np.sin(drive * times)[None, :]
    )
    return homogeneous + particular


def _rising_state(instant, drive, decay_rate, damped, static_half, cosine, sine, cosine_state, sine_state):
    """Displacement and velocity at one instant inside the ramp."""
    decay = np.exp(-decay_rate * instant)
    angle = damped * instant
    homogeneous = decay * (cosine_state * np.cos(angle) + sine_state * np.sin(angle))
    homogeneous_velocity = -decay_rate * homogeneous + decay * damped * (
        -cosine_state * np.sin(angle) + sine_state * np.cos(angle)
    )
    displacement = homogeneous + static_half + cosine * np.cos(drive * instant) + sine * np.sin(drive * instant)
    velocity = (
        homogeneous_velocity
        - cosine * drive * np.sin(drive * instant)
        + sine * drive * np.cos(drive * instant)
    )
    return displacement, velocity


def report_tip_errors(reference, results) -> None:
    print("RMS tip-displacement error over the time window (m)")
    header = (
        f"{'modes':>6}  {'dofs CB':>8}  {'dofs Rubin':>11}  "
        f"{'x CB':>10}  {'y CB':>10}  {'x Rubin':>10}  {'y Rubin':>10}"
    )
    print(header)
    print("-" * len(header))
    for n_modes in MODE_COUNTS:
        craig, rubin_history, craig_dofs, rubin_dofs = results[n_modes]
        craig_error = _rms(craig - reference)
        rubin_error = _rms(rubin_history - reference)
        print(
            f"{n_modes:6d}  {craig_dofs:8d}  {rubin_dofs:11d}  "
            f"{craig_error[0]:10.3e}  {craig_error[1]:10.3e}  "
            f"{rubin_error[0]:10.3e}  {rubin_error[1]:10.3e}"
        )
    reference_norm = _tip_norm(reference)
    print("Relative L2 error of the tip-norm history (%)")
    header = f"{'modes':>6}  {'Craig-Bampton':>14}  {'Rubin':>14}"
    print(header)
    print("-" * len(header))
    for n_modes in MODE_COUNTS:
        craig, rubin_history, _, _ = results[n_modes]
        print(
            f"{n_modes:6d}  "
            f"{_relative_l2_percent(reference_norm, _tip_norm(craig)):14.3f}  "
            f"{_relative_l2_percent(reference_norm, _tip_norm(rubin_history)):14.3f}"
        )


def tip_history_figure(times, reference, results):
    """Horizontal and vertical tip displacement for each elastic-mode count."""
    figure, axes = plt.subplots(
        2,
        len(MODE_COUNTS),
        figsize=(12.8, 5.6),
        sharex=True,
        squeeze=False,
    )
    milliseconds = 1e3 * times
    for column, n_modes in enumerate(MODE_COUNTS):
        craig, rubin_history, _, _ = results[n_modes]
        for row, (name, full, craig_row, rubin_row) in enumerate(
            (
                ("x", reference[0], craig[0], rubin_history[0]),
                ("y", reference[1], craig[1], rubin_history[1]),
            )
        ):
            axis = axes[row, column]
            axis.plot(milliseconds, 1e3 * full, color="black", linewidth=1.6, label="FOM")
            axis.plot(milliseconds, 1e3 * rubin_row, color="C0", linewidth=1.2, label="Rubin")
            axis.plot(
                milliseconds,
                1e3 * craig_row,
                color="C1",
                linestyle="--",
                linewidth=1.2,
                label="Craig-Bampton",
            )
            axis.grid(True, alpha=0.3)
            if row == 0:
                axis.set_title(f"{n_modes} elastic modes")
            if column == 0:
                axis.set_ylabel(f"Tip {name} (mm)")
            if row == 1:
                axis.set_xlabel("Time (ms)")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.tight_layout()
    figure.subplots_adjust(top=0.84)
    figure.suptitle(
        f"{TIP_FORCE / 1000:.0f} kN held at the upper tip, {N_SUBSTRUCTURES} substructures"
    )
    figure.legend(handles, labels, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 0.94))
    output = ROOT / "figures" / "tip_history.png"
    output.parent.mkdir(exist_ok=True)
    figure.savefig(output, dpi=150)
    plt.close(figure)
    return output


def tip_norm_error_figure(reference, results):
    """Relative L2 error of the tip-displacement norm as elastic modes are added."""
    reference_norm = _tip_norm(reference)
    rubin_error = []
    craig_error = []
    for n_modes in MODE_COUNTS:
        craig, rubin_history, _, _ = results[n_modes]
        craig_error.append(_relative_l2_percent(reference_norm, _tip_norm(craig)))
        rubin_error.append(_relative_l2_percent(reference_norm, _tip_norm(rubin_history)))
    figure, axis = plt.subplots(figsize=(6.4, 4.2))
    axis.plot(MODE_COUNTS, rubin_error, color="C0", linestyle="-", marker="o", label="Rubin")
    axis.plot(MODE_COUNTS, craig_error, color="C1", linestyle="--", marker="s", label="Craig-Bampton")
    axis.set_xticks(list(MODE_COUNTS))
    axis.set_xlabel("Elastic modes per substructure")
    axis.set_ylabel("Relative L2 error (%)")
    axis.grid(True, alpha=0.3)
    axis.legend()
    figure.tight_layout()
    output = ROOT / "figures" / "tip_norm_error.png"
    output.parent.mkdir(exist_ok=True)
    figure.savefig(output, dpi=150)
    plt.close(figure)
    return output


def reduced_force(component: ReducedModel, force, coupled: CoupledModel, component_index: int):
    """Project a force on one component into the coupled reduced coordinates."""
    local_force = component.basis.T @ force
    assembled = np.zeros(coupled.stiffness.shape[0])
    assembled[coupled.boundary[component_index]] = local_force[: component.n_boundary]
    assembled[coupled.local[component_index]] = local_force[component.n_boundary :]
    return assembled


def recovery_row(component: ReducedModel, dof: int, coupled: CoupledModel, component_index: int):
    """Row that recovers one finite-element unknown from the coupled coordinates."""
    row = np.zeros(coupled.stiffness.shape[0])
    row[coupled.boundary[component_index]] = component.basis[dof, : component.n_boundary]
    row[coupled.local[component_index]] = component.basis[dof, component.n_boundary :]
    return row


def tip_dofs(model) -> tuple[int, int]:
    """Horizontal and vertical unknowns at the upper free corner."""
    vertical = np.flatnonzero(model.component == 1)
    coordinates = model.coordinates[vertical]
    distance = (coordinates[:, 0] - LENGTH) ** 2 + (coordinates[:, 1] - HEIGHT) ** 2
    vertical_dof = int(vertical[np.argmin(distance)])
    target = model.coordinates[vertical_dof]
    horizontal = np.flatnonzero(model.component == 0)
    match = np.all(np.isclose(model.coordinates[horizontal], target, atol=1e-10), axis=1)
    if not np.any(match):
        raise RuntimeError("The upper free corner has no horizontal unknown")
    return int(horizontal[np.argmax(match)]), vertical_dof


def frequencies(stiffness, mass, count: int) -> NDArray[np.float64]:
    values = eigh(stiffness, mass, eigvals_only=True, subset_by_index=[0, count - 1])
    return np.sqrt(np.maximum(values, 0.0)) / (2.0 * np.pi)


def _rms(values) -> NDArray[np.float64]:
    return np.sqrt(np.mean(values**2, axis=1))


def _tip_norm(history) -> NDArray[np.float64]:
    """Euclidean norm of the tip displacement at each sample."""
    return np.sqrt(np.sum(history**2, axis=0))


def _relative_l2_percent(reference_norm, approximation_norm) -> float:
    """Relative L2 difference of two tip-norm histories, in percent."""
    scale = np.linalg.norm(reference_norm)
    return float(100.0 * np.linalg.norm(approximation_norm - reference_norm) / scale)


if __name__ == "__main__":
    main()
