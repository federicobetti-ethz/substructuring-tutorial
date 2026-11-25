"""Craig–Bampton, Rubin, and static condensation of a plane-stress cantilever."""

import sys
from pathlib import Path

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
    plot_tip,
    rectangle_mesh,
    rubin,
    static_condensation,
    structural_model,
)
from substructuring.coupling import CoupledModel, ReducedModel  # noqa: E402

LENGTH = 2.0
HEIGHT = 0.35
NX = 40
NY = 10
N_SUBSTRUCTURES = 2
MODE_COUNTS = (2, 4, 6, 8, 12, 16, 24, 32)
HISTORY_MODES = 16
DAMPING_RATIO = 0.01
CYCLES = 6.0
RAMP_CYCLES = 0.08
TIP_FORCE = 10_000.0
N_TIMES = 2000
FREQUENCY_BAND_HZ = 1700.0
MATERIAL = Material(young=210.0e9, poisson=0.3, density=7800.0, thickness=0.01)


def main() -> None:
    check_craig_bampton_is_complete()
    mesh = rectangle_mesh(0.0, LENGTH, HEIGHT, NX, NY)
    full = structural_model(mesh, MATERIAL, clamp="left")
    fundamental = frequencies(full.stiffness, full.mass, 1)[0]
    ramp = RAMP_CYCLES / fundamental
    times = np.linspace(0.0, CYCLES / fundamental, N_TIMES)
    print(f"Free degrees of freedom in the full model: {full.stiffness.shape[0]}")
    print(
        f"{TIP_FORCE:.0f} N in x and y, raised over {1e3 * ramp:.2f} ms and then held, "
        f"followed until {1e3 * times[-1]:.1f} ms"
    )
    reference = tip_history(
        full.stiffness, full.mass, _tip_force(full), _tip_rows(full), times, ramp
    )
    tip_marker = plot_tip(mesh, ROOT / "figures" / "tip.png")
    components = beam_components(N_SUBSTRUCTURES, NX, NY)
    tip = components[-1]
    print(
        f"{N_SUBSTRUCTURES} substructures, {tip.boundary.size} degrees of freedom per cut"
    )
    condensed_parts = [
        static_condensation(component.stiffness, component.mass, component.boundary)
        for component in components
    ]
    condensed = couple(condensed_parts)
    condensed_history = component_history(
        tip, condensed_parts[-1], condensed, times, ramp
    )
    print(f"Static condensation, reduced size {condensed.stiffness.shape[0]}")
    results = {}
    reduced_frequencies = {}
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
        reduced_frequencies[n_modes] = (
            _frequencies_in_band(craig.stiffness, craig.mass, FREQUENCY_BAND_HZ),
            _frequencies_in_band(
                rubin_model.stiffness, rubin_model.mass, FREQUENCY_BAND_HZ
            ),
        )
    frequency_errors = _successive_frequency_changes(reduced_frequencies)
    report_tip_errors(reference, results, condensed_history)
    report_frequency_errors(reduced_frequencies, frequency_errors)
    history = tip_history_figure(
        times, reference, results[HISTORY_MODES], condensed_history
    )
    condensation = condensation_history_figure(times, reference, condensed_history)
    convergence = tip_norm_error_figure(reference, results)
    error = tip_error_figure(reference, results)
    frequency_figure = frequency_convergence_figure(
        reduced_frequencies, frequency_errors
    )
    print(f"Wrote {tip_marker}")
    print(f"Wrote {history}")
    print(f"Wrote {condensation}")
    print(f"Wrote {convergence}")
    print(f"Wrote {error}")
    print(f"Wrote {frequency_figure}")


def beam_components(n_sub: int, nx: int, ny: int):
    """Clamp the root and cut the beam into ``n_sub`` equal pieces."""
    if n_sub < 2:
        raise ValueError("A chain needs at least two substructures")
    if nx % n_sub:
        raise ValueError(
            f"{nx} elements along the beam do not divide into {n_sub} pieces"
        )
    nx_piece = nx // n_sub
    stations = np.linspace(0.0, LENGTH, n_sub + 1)
    components = []
    for index in range(n_sub):
        mesh = rectangle_mesh(
            stations[index], stations[index + 1], HEIGHT, nx_piece, ny
        )
        if index == 0:
            interface = "right"
            clamp = "left"
        elif index == n_sub - 1:
            interface = "left"
            clamp = None
        else:
            interface = ("left", "right")
            clamp = None
        components.append(
            structural_model(mesh, MATERIAL, interface=interface, clamp=clamp)
        )
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
            raise RuntimeError(
                f"Cut {index - 1} does not have matching degrees of freedom"
            )
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
            craig_bampton(
                component.stiffness, component.mass, component.boundary, n_modes
            )
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
    force = reduced_force(
        reduced, _tip_load(tip, horizontal, vertical), coupled, component_index
    )
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
        times,
        drive,
        decay_rate,
        damped,
        static_half,
        cosine,
        sine,
        cosine_state,
        sine_state,
    )
    end = _rising_state(
        ramp,
        drive,
        decay_rate,
        damped,
        static_half,
        cosine,
        sine,
        cosine_state,
        sine_state,
    )
    held = modal_force / natural
    held_cosine = end[0] - held
    held_sine = (end[1] + decay_rate * held_cosine) / damped
    elapsed = times - ramp
    settled = held[:, None] + np.exp(
        -decay_rate[:, None] * np.maximum(elapsed, 0.0)[None, :]
    ) * (
        held_cosine[:, None] * np.cos(damped[:, None] * elapsed[None, :])
        + held_sine[:, None] * np.sin(damped[:, None] * elapsed[None, :])
    )
    return np.where(times[None, :] <= ramp, rising, settled)


def _rising_response(
    times,
    drive,
    decay_rate,
    damped,
    static_half,
    cosine,
    sine,
    cosine_state,
    sine_state,
):
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


def _rising_state(
    instant,
    drive,
    decay_rate,
    damped,
    static_half,
    cosine,
    sine,
    cosine_state,
    sine_state,
):
    """Displacement and velocity at one instant inside the ramp."""
    decay = np.exp(-decay_rate * instant)
    angle = damped * instant
    homogeneous = decay * (cosine_state * np.cos(angle) + sine_state * np.sin(angle))
    homogeneous_velocity = -decay_rate * homogeneous + decay * damped * (
        -cosine_state * np.sin(angle) + sine_state * np.cos(angle)
    )
    displacement = (
        homogeneous
        + static_half
        + cosine * np.cos(drive * instant)
        + sine * np.sin(drive * instant)
    )
    velocity = (
        homogeneous_velocity
        - cosine * drive * np.sin(drive * instant)
        + sine * drive * np.cos(drive * instant)
    )
    return displacement, velocity


def report_tip_errors(reference, results, condensed_history) -> None:
    condensed_error = _rms(condensed_history - reference)
    condensed_percent = _relative_l2_percent(
        _tip_norm(reference), _tip_norm(condensed_history)
    )
    print(
        "Static condensation RMS tip error "
        f"x {condensed_error[0]:.3e} m, y {condensed_error[1]:.3e} m, "
        f"relative L2 {condensed_percent:.3f} %"
    )
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


def tip_history_figure(times, reference, comparison, condensed_history):
    """Tip displacement at :data:`HISTORY_MODES` elastic modes per piece."""
    craig, rubin_history, _, _ = comparison
    figure, axes = plt.subplots(2, 1, figsize=(7.4, 5.4), sharex=True)
    milliseconds = 1e3 * times
    for axis, name, full, craig_row, rubin_row, condensed_row in (
        (axes[0], "x", reference[0], craig[0], rubin_history[0], condensed_history[0]),
        (axes[1], "y", reference[1], craig[1], rubin_history[1], condensed_history[1]),
    ):
        axis.plot(milliseconds, 1e3 * full, color="black", linewidth=1.6, label="FOM")
        axis.plot(
            milliseconds,
            1e3 * condensed_row,
            color="C2",
            linestyle=":",
            linewidth=1.5,
            label="Static condensation",
        )
        axis.plot(
            milliseconds,
            1e3 * rubin_row,
            color="C0",
            linestyle="--",
            linewidth=1.3,
            label="Rubin",
        )
        axis.plot(
            milliseconds,
            1e3 * craig_row,
            color="C1",
            linestyle="--",
            linewidth=1.3,
            label="Craig-Bampton",
        )
        axis.set_ylabel(f"Tip {name} (mm)")
        axis.grid(True, alpha=0.3)
    axes[1].set_xlabel("Time (ms)")
    handles, labels = axes[0].get_legend_handles_labels()
    figure.tight_layout()
    figure.subplots_adjust(top=0.86)
    figure.suptitle(f"{HISTORY_MODES} elastic modes, {N_SUBSTRUCTURES} substructures")
    figure.legend(
        handles,
        labels,
        loc="upper center",
        ncol=4,
        frameon=False,
        bbox_to_anchor=(0.5, 0.96),
    )
    output = ROOT / "figures" / "tip_history.png"
    output.parent.mkdir(exist_ok=True)
    figure.savefig(output, dpi=150)
    plt.close(figure)
    return output


def condensation_history_figure(times, reference, condensed_history):
    """Tip displacement of the full model and of static condensation."""
    figure, axes = plt.subplots(2, 1, figsize=(7.2, 5.2), sharex=True)
    milliseconds = 1e3 * times
    for axis, name, full, condensed in (
        (axes[0], "x", reference[0], condensed_history[0]),
        (axes[1], "y", reference[1], condensed_history[1]),
    ):
        axis.plot(milliseconds, 1e3 * full, color="black", linewidth=1.6, label="FOM")
        axis.plot(
            milliseconds,
            1e3 * condensed,
            color="C2",
            linestyle=":",
            linewidth=1.6,
            label="Static condensation",
        )
        axis.set_ylabel(f"Tip {name} (mm)")
        axis.grid(True, alpha=0.3)
    axes[1].set_xlabel("Time (ms)")
    handles, labels = axes[0].get_legend_handles_labels()
    figure.tight_layout()
    figure.subplots_adjust(top=0.88)
    figure.suptitle(
        f"{TIP_FORCE / 1000:.0f} kN held at the upper tip, {N_SUBSTRUCTURES} substructures"
    )
    figure.legend(
        handles,
        labels,
        loc="upper center",
        ncol=2,
        frameon=False,
        bbox_to_anchor=(0.5, 0.96),
    )
    output = ROOT / "figures" / "tip_condensation.png"
    output.parent.mkdir(exist_ok=True)
    figure.savefig(output, dpi=150)
    plt.close(figure)
    return output


def tip_error_figure(reference, results):
    """Relative tip error in x and y as elastic modes are added.

    The error is the relative :math:`L^2` difference from the full model, in
    percent.
    """
    rubin_error = []
    craig_error = []
    for n_modes in MODE_COUNTS:
        craig, rubin_history, _, _ = results[n_modes]
        rubin_error.append(_history_errors(reference, rubin_history)[:2])
        craig_error.append(_history_errors(reference, craig)[:2])
    rubin_error = np.asarray(rubin_error)
    craig_error = np.asarray(craig_error)
    figure, axes = plt.subplots(2, 1, figsize=(6.6, 5.4), sharex=True)
    for axis, name, rubin_row, craig_row in zip(
        axes, ("Tip x", "Tip y"), rubin_error.T, craig_error.T
    ):
        axis.plot(
            MODE_COUNTS,
            rubin_row,
            color="C0",
            linestyle="--",
            marker="o",
            label="Rubin",
        )
        axis.plot(
            MODE_COUNTS,
            craig_row,
            color="C1",
            linestyle="--",
            marker="s",
            label="Craig-Bampton",
        )
        axis.set_ylabel(f"{name} error (%)")
        axis.grid(True, alpha=0.3)
    axes[-1].set_xticks(list(MODE_COUNTS))
    axes[-1].set_xlabel("Elastic modes per substructure")
    handles, labels = axes[0].get_legend_handles_labels()
    figure.tight_layout()
    figure.subplots_adjust(top=0.90)
    figure.legend(
        handles,
        labels,
        loc="upper center",
        ncol=2,
        frameon=False,
        bbox_to_anchor=(0.5, 0.98),
    )
    output = ROOT / "figures" / "tip_error.png"
    output.parent.mkdir(exist_ok=True)
    figure.savefig(output, dpi=150)
    plt.close(figure)
    return output


def report_frequency_errors(reduced_frequencies, frequency_errors) -> None:
    print(
        f"Maximum frequency change between successive refinements, up to {FREQUENCY_BAND_HZ:.0f} Hz (%)"
    )
    header = f"{'step':>10}  {'Craig-Bampton':>14}  {'Rubin':>14}"
    print(header)
    print("-" * len(header))
    for previous, n_modes in zip(MODE_COUNTS, MODE_COUNTS[1:]):
        craig_error, rubin_error = frequency_errors[n_modes]
        print(
            f"{previous:4d} -> {n_modes:<4d}  {np.max(craig_error):14.3f}  {np.max(rubin_error):14.3f}"
        )
    richest = MODE_COUNTS[-1]
    craig_reference, rubin_reference = reduced_frequencies[richest]
    print(
        f"Craig-Bampton frequencies at {richest} modes (Hz): "
        + _format_hertz(craig_reference)
    )
    print(
        f"Rubin frequencies at {richest} modes (Hz): " + _format_hertz(rubin_reference)
    )


def _successive_frequency_changes(reduced_frequencies):
    """Relative change of each frequency from one mode count to the next."""
    changes = {}
    previous = None
    for n_modes in MODE_COUNTS:
        if previous is None:
            previous = n_modes
            continue
        pair = []
        for index in range(2):
            before = reduced_frequencies[previous][index]
            after = reduced_frequencies[n_modes][index]
            count = min(before.size, after.size)
            if count == 0:
                raise RuntimeError("The target band contains no reduced frequencies")
            pair.append(100.0 * np.abs(after[:count] - before[:count]) / after[:count])
        changes[n_modes] = tuple(pair)
        previous = n_modes
    return changes


def _format_hertz(values) -> str:
    return ", ".join(f"{value:.1f}" for value in values)


def frequency_convergence_figure(reduced_frequencies, frequency_errors):
    """Change of each reduced frequency between successive mode counts.

    A point at ``n`` modes is the relative change from the previous count. The
    color is that frequency of the finest model, from the lowest upward.
    """
    richest = MODE_COUNTS[-1]
    compared = MODE_COUNTS[1:]
    figure, axes = plt.subplots(2, 1, figsize=(7.2, 7.2), sharex=True, sharey=True)
    for axis, name, index in ((axes[0], "Craig-Bampton", 0), (axes[1], "Rubin", 1)):
        reference = reduced_frequencies[richest][index]
        colors = plt.cm.viridis(np.linspace(0.1, 0.9, reference.size))
        for mode, (frequency, color) in enumerate(zip(reference, colors)):
            counts = [
                n_modes
                for n_modes in compared
                if mode < frequency_errors[n_modes][index].size
            ]
            error = np.array(
                [frequency_errors[n_modes][index][mode] for n_modes in counts]
            )
            axis.plot(
                counts,
                np.where(error > 0.0, error, np.nan),
                color=color,
                marker="o",
                linewidth=1.2,
                label=f"{frequency:.0f} Hz",
            )
        axis.set_yscale("log")
        axis.set_title(name)
        axis.set_ylabel("Change from the previous refinement (%)")
        axis.grid(True, alpha=0.3, which="both")
        axis.legend(frameon=False, fontsize=8, loc="upper right")
    axes[1].set_xticks(list(MODE_COUNTS))
    axes[1].set_xlim(MODE_COUNTS[0] - 1, MODE_COUNTS[-1] + 1)
    axes[1].set_xlabel("Elastic modes per substructure")
    figure.tight_layout()
    output = ROOT / "figures" / "frequency_convergence.png"
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
        rubin_error.append(
            _relative_l2_percent(reference_norm, _tip_norm(rubin_history))
        )
    figure, axis = plt.subplots(figsize=(6.4, 4.2))
    axis.plot(
        MODE_COUNTS, rubin_error, color="C0", linestyle="-", marker="o", label="Rubin"
    )
    axis.plot(
        MODE_COUNTS,
        craig_error,
        color="C1",
        linestyle="--",
        marker="s",
        label="Craig-Bampton",
    )
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


def reduced_force(
    component: ReducedModel, force, coupled: CoupledModel, component_index: int
):
    """Project a force on one component into the coupled reduced coordinates."""
    local_force = component.basis.T @ force
    assembled = np.zeros(coupled.stiffness.shape[0])
    assembled[coupled.boundary[component_index]] = local_force[: component.n_boundary]
    assembled[coupled.local[component_index]] = local_force[component.n_boundary :]
    return assembled


def recovery_row(
    component: ReducedModel, dof: int, coupled: CoupledModel, component_index: int
):
    """Row that recovers one finite-element unknown from the coupled coordinates."""
    row = np.zeros(coupled.stiffness.shape[0])
    row[coupled.boundary[component_index]] = component.basis[
        dof, : component.n_boundary
    ]
    row[coupled.local[component_index]] = component.basis[dof, component.n_boundary :]
    return row


def tip_dofs(model) -> tuple[int, int]:
    """Horizontal and vertical unknowns at the upper free corner."""
    return _node_dofs(model, LENGTH, HEIGHT)


def _node_dofs(model, x: float, y: float) -> tuple[int, int]:
    vertical = np.flatnonzero(model.component == 1)
    coordinates = model.coordinates[vertical]
    distance = (coordinates[:, 0] - x) ** 2 + (coordinates[:, 1] - y) ** 2
    vertical_dof = int(vertical[np.argmin(distance)])
    target = model.coordinates[vertical_dof]
    if not np.allclose(target, (x, y), atol=1e-8):
        raise RuntimeError(f"No node at ({x}, {y})")
    horizontal = np.flatnonzero(model.component == 0)
    match = np.all(
        np.isclose(model.coordinates[horizontal], target, atol=1e-10), axis=1
    )
    if not np.any(match):
        raise RuntimeError(f"The node at ({x}, {y}) has no horizontal unknown")
    return int(horizontal[np.argmax(match)]), vertical_dof


def _frequencies_in_band(stiffness, mass, upper_hz: float) -> NDArray[np.float64]:
    """Lowest natural frequencies that lie at or below ``upper_hz``."""
    count = min(80, stiffness.shape[0])
    values = frequencies(stiffness, mass, count)
    inside = values[values <= upper_hz]
    if inside.size == count and count < stiffness.shape[0]:
        raise RuntimeError(
            "The frequency band still contains the highest computed mode"
        )
    if inside.size == 0:
        raise RuntimeError("The target band contains no natural frequencies")
    return inside


def _frequency_error_percent(reference, stiffness, mass) -> NDArray[np.float64]:
    """Relative error of the lowest frequencies, paired in order, in percent."""
    if stiffness.shape[0] < reference.size:
        raise RuntimeError(
            "The reduced model has fewer frequencies than the target band"
        )
    obtained = frequencies(stiffness, mass, reference.size)
    return 100.0 * np.abs(obtained - reference) / reference


def frequencies(stiffness, mass, count: int) -> NDArray[np.float64]:
    values = eigh(stiffness, mass, eigvals_only=True, subset_by_index=[0, count - 1])
    return np.sqrt(np.maximum(values, 0.0)) / (2.0 * np.pi)


def _rms(values) -> NDArray[np.float64]:
    return np.sqrt(np.mean(values**2, axis=1))


def _tip_norm(history) -> NDArray[np.float64]:
    """Euclidean norm of the tip displacement at each sample."""
    return np.sqrt(np.sum(history**2, axis=0))


def _history_errors(reference, approximation) -> tuple[float, float, float]:
    """Relative L2 error of tip x, tip y, and the tip norm, in percent."""
    return (
        _relative_l2_percent(reference[0], approximation[0]),
        _relative_l2_percent(reference[1], approximation[1]),
        _relative_l2_percent(_tip_norm(reference), _tip_norm(approximation)),
    )


def _relative_l2_percent(reference_norm, approximation_norm) -> float:
    """Relative L2 difference of two tip-norm histories, in percent."""
    scale = np.linalg.norm(reference_norm)
    return float(100.0 * np.linalg.norm(approximation_norm - reference_norm) / scale)


if __name__ == "__main__":
    main()
