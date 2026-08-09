from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np


POWER_ATOL = 1.0e-8
ANGLE_ATOL_DEG = 1.0e-10


FORWARD_REVERSE_DTYPE = np.dtype([
    ("lambda_um", "f8"),
    ("pol", "U4"),
    ("theta_deg", "f8"),
    ("phi_deg", "f8"),
    ("harmonics", "i4"),
    ("R", "f8"),
    ("T", "f8"),
    ("A", "f8"),
    ("R_reverse", "f8"),
    ("T_reverse", "f8"),
    ("A_reverse", "f8"),
    ("emissivity_same_channel", "f8"),
    ("emissivity_reverse_channel", "f8"),
    ("eta", "f8"),
    ("conservation", "f8"),
    ("reverse_conservation", "f8"),
])


@dataclass(frozen=True)
class PeakMetrics:
    """Sub-grid peak location and sampled-data full width at half maximum."""

    wavelength_um: float
    value: float
    fwhm_um: float
    left_half_max_um: float
    right_half_max_um: float


def measure_peak(
    wavelengths_um: Sequence[float],
    values: Sequence[float],
    *,
    window_um: tuple[float, float] | None = None,
) -> PeakMetrics:
    """Measure one strongest peak using a quadratic vertex and linear FWHM.

    The linewidth is the conventional full width at half of the peak value,
    without background subtraction.  A peak on the analysis-window boundary,
    or a peak whose two half-maximum crossings are not sampled, is rejected
    instead of returning a misleading linewidth.
    """
    wavelength = np.asarray(wavelengths_um, dtype=float)
    observable = np.asarray(values, dtype=float)
    if wavelength.ndim != 1 or observable.ndim != 1:
        raise ValueError("wavelengths_um and values must be one-dimensional")
    if len(wavelength) != len(observable) or len(wavelength) < 3:
        raise ValueError("wavelengths_um and values must have equal length >= 3")
    if not np.all(np.isfinite(wavelength)) or not np.all(np.isfinite(observable)):
        raise ValueError("peak inputs must contain only finite values")
    if np.any(np.diff(wavelength) <= 0.0):
        raise ValueError("wavelengths_um must be strictly increasing")

    if window_um is None:
        first = 0
        last = len(wavelength) - 1
    else:
        lower_um, upper_um = (float(value) for value in window_um)
        if not np.isfinite(lower_um) or not np.isfinite(upper_um) or lower_um >= upper_um:
            raise ValueError("window_um must be a finite increasing pair")
        selected = np.flatnonzero(
            (wavelength >= lower_um) & (wavelength <= upper_um)
        )
        if len(selected) < 3:
            raise ValueError("window_um must contain at least three wavelength samples")
        first = int(selected[0])
        last = int(selected[-1])

    peak_index = first + int(np.argmax(observable[first : last + 1]))
    if peak_index == first or peak_index == last:
        raise ValueError("strongest peak lies on the analysis-window boundary")

    x0 = float(wavelength[peak_index])
    local_x = wavelength[peak_index - 1 : peak_index + 2] - x0
    local_y = observable[peak_index - 1 : peak_index + 2]
    quadratic, linear, constant = np.polyfit(local_x, local_y, 2)
    peak_offset = 0.0
    peak_value = float(observable[peak_index])
    if quadratic < 0.0:
        candidate_offset = float(-linear / (2.0 * quadratic))
        if float(local_x[0]) <= candidate_offset <= float(local_x[-1]):
            candidate_value = float(
                quadratic * candidate_offset**2
                + linear * candidate_offset
                + constant
            )
            if candidate_value >= peak_value:
                peak_offset = candidate_offset
                peak_value = candidate_value
    if peak_value <= 0.0:
        raise ValueError("strongest peak must have a positive value")

    half_maximum = 0.5 * peak_value
    left_candidates = np.flatnonzero(
        observable[first:peak_index] <= half_maximum
    )
    right_candidates = np.flatnonzero(
        observable[peak_index + 1 : last + 1] <= half_maximum
    )
    if len(left_candidates) == 0 or len(right_candidates) == 0:
        raise ValueError(
            "analysis window does not contain both half-maximum crossings"
        )

    left_low = first + int(left_candidates[-1])
    left_high = left_low + 1
    right_high = peak_index + 1 + int(right_candidates[0])
    right_low = right_high - 1

    def crossing(index_a: int, index_b: int) -> float:
        value_a = float(observable[index_a])
        value_b = float(observable[index_b])
        if value_a == value_b:
            raise ValueError("cannot interpolate a flat half-maximum crossing")
        fraction = (half_maximum - value_a) / (value_b - value_a)
        if fraction < -1.0e-12 or fraction > 1.0 + 1.0e-12:
            raise ValueError("half-maximum crossing is not bracketed")
        return float(
            wavelength[index_a]
            + fraction * (wavelength[index_b] - wavelength[index_a])
        )

    left_crossing = crossing(left_low, left_high)
    right_crossing = crossing(right_high, right_low)
    width_um = right_crossing - left_crossing
    if width_um <= 0.0:
        raise ValueError("interpolated FWHM must be positive")
    return PeakMetrics(
        wavelength_um=x0 + peak_offset,
        value=peak_value,
        fwhm_um=width_um,
        left_half_max_um=left_crossing,
        right_half_max_um=right_crossing,
    )


def solve_forward_reverse_spectrum(
    *,
    rcwa: object,
    build_simulation: Callable[[object], object],
    wavelengths_um: Sequence[float],
    polarizations: Sequence[str],
    angles_deg: Sequence[float],
    phi_deg: float,
    workers_per_angle: int = 0,
) -> np.ndarray:
    """Solve an ordered forward/reverse angle pair and return one paired table.

    An incident ray labelled ``+theta`` is time-reversed by the outgoing row
    in the ``-theta`` Bloch sector.  Consequently
    ``emissivity_same_channel`` is the exact row deficiency from the reverse
    solve, not an assumed ``1-R(-theta)`` surrogate.

    ``T`` contains transmission into radiative exterior channels and ``A``
    contains total thermodynamic absorptivity.  Power entering a lossy
    exterior half-space is therefore included in ``A`` rather than ``T``.

    ``workers_per_angle=0`` uses the backend's hardware-limited automatic
    concurrency.  A positive value sets an explicit per-angle worker count.
    """
    wavelength_values = np.asarray(wavelengths_um, dtype=float)
    if wavelength_values.ndim != 1 or len(wavelength_values) == 0:
        raise ValueError("wavelengths_um must be a nonempty one-dimensional sequence")
    if not np.all(np.isfinite(wavelength_values)) or np.any(wavelength_values <= 0.0):
        raise ValueError("wavelengths_um must contain finite positive values")
    if len(np.unique(wavelength_values)) != len(wavelength_values):
        raise ValueError("wavelengths_um must not contain duplicates")
    polarization_values = tuple(str(value) for value in polarizations)
    if len(polarization_values) == 0 or len(set(polarization_values)) != len(
        polarization_values
    ):
        raise ValueError("polarizations must be nonempty and unique")
    if len(angles_deg) != 2:
        raise ValueError("angles_deg must contain the forward and reverse angles")
    forward_angle_deg = float(angles_deg[0])
    reverse_angle_deg = float(angles_deg[1])
    if (
        not np.isfinite(forward_angle_deg)
        or not np.isfinite(reverse_angle_deg)
        or
        forward_angle_deg * reverse_angle_deg >= 0.0
        or not np.isclose(
            abs(forward_angle_deg),
            abs(reverse_angle_deg),
            rtol=0.0,
            atol=1e-12,
        )
    ):
        raise ValueError(
            "angles_deg must be an ordered pair of nonzero opposite angles"
        )
    phi_value_deg = float(phi_deg)
    if not np.isfinite(phi_value_deg):
        raise ValueError("phi_deg must be finite")
    if isinstance(workers_per_angle, (bool, np.bool_)) or int(workers_per_angle) != workers_per_angle:
        raise ValueError("workers_per_angle must be a nonnegative integer")
    workers_per_angle_value = int(workers_per_angle)
    if workers_per_angle_value < 0:
        raise ValueError("workers_per_angle must be a nonnegative integer")

    sim = build_simulation(rcwa)
    if not hasattr(sim, "GetSpectrumAndDirectionalThermalChannelsForAngles"):
        raise RuntimeError(
            "rcwa backend must expose the combined exact spectrum/thermal API"
        )

    workers = (
        0
        if workers_per_angle_value == 0
        else workers_per_angle_value * len(angles_deg)
    )
    raw, thermal_channels = (
        sim.GetSpectrumAndDirectionalThermalChannelsForAngles(
            wavelength_values,
            angles_deg,
            polarization_values,
            Phi=phi_value_deg,
            Workers=workers,
        )
    )
    forward_channels = thermal_channels[np.isclose(
        thermal_channels["theta_deg"],
        forward_angle_deg,
        rtol=0.0,
        atol=ANGLE_ATOL_DEG,
    )]
    reverse_channels = thermal_channels[np.isclose(
        thermal_channels["theta_deg"],
        reverse_angle_deg,
        rtol=0.0,
        atol=ANGLE_ATOL_DEG,
    )]
    return combine_forward_reverse_spectrum(
        raw,
        forward_channels,
        reverse_channels,
        forward_angle_deg=forward_angle_deg,
        reverse_angle_deg=reverse_angle_deg,
    )


def _try_combine_forward_reverse_spectrum_fast(
    raw: np.ndarray,
    forward_channels: np.ndarray,
    reverse_channels: np.ndarray,
    forward_angle_deg: float,
    reverse_angle_deg: float,
) -> np.ndarray | None:
    row_count = len(forward_channels)
    if len(reverse_channels) != row_count or len(raw) != 2 * row_count:
        return None
    if row_count == 0:
        return np.empty(0, dtype=FORWARD_REVERSE_DTYPE)
    if any(
        rows.dtype.fields["pol"][0].kind != "U"
        for rows in (raw, forward_channels, reverse_channels)
    ):
        return None

    raw_forward = raw[:row_count]
    raw_reverse = raw[row_count:]
    angle_groups = (
        (raw_forward, forward_angle_deg),
        (raw_reverse, reverse_angle_deg),
        (forward_channels, forward_angle_deg),
        (reverse_channels, reverse_angle_deg),
    )
    if any(
        not np.all(np.isclose(
            rows["theta_deg"],
            expected,
            rtol=0.0,
            atol=ANGLE_ATOL_DEG,
        ))
        for rows, expected in angle_groups
    ):
        return None

    def sorted_rows(rows: np.ndarray) -> np.ndarray:
        order = np.lexsort((rows["pol"], rows["lambda_um"]))
        return rows[order]

    forward = sorted_rows(raw_forward)
    reverse = sorted_rows(raw_reverse)
    absorption_rows = sorted_rows(forward_channels)
    emissivity_rows = sorted_rows(reverse_channels)

    def same_keys(left: np.ndarray, right: np.ndarray) -> bool:
        return bool(
            np.array_equal(left["lambda_um"], right["lambda_um"])
            and np.array_equal(left["pol"], right["pol"])
        )

    duplicate_keys = (
        (forward["lambda_um"][1:] == forward["lambda_um"][:-1])
        & (forward["pol"][1:] == forward["pol"][:-1])
    )
    if np.any(duplicate_keys) or not all(
        same_keys(forward, rows)
        for rows in (reverse, absorption_rows, emissivity_rows)
    ):
        return None

    try:
        forward_transmission = (
            np.asarray(absorption_rows["incident_scattering"], dtype=float)
            - np.asarray(forward["R"], dtype=float)
        )
        reverse_transmission = (
            np.asarray(emissivity_rows["incident_scattering"], dtype=float)
            - np.asarray(reverse["R"], dtype=float)
        )
        forward_absorption = np.asarray(
            absorption_rows["absorptivity"], dtype=float
        )
        reverse_absorption = np.asarray(
            emissivity_rows["absorptivity"], dtype=float
        )
        bounded_power = np.column_stack((
            forward["R"], forward["T"], forward["A"],
            absorption_rows["absorptivity"], absorption_rows["emissivity"],
            reverse["R"], reverse["T"], reverse["A"],
            emissivity_rows["absorptivity"], emissivity_rows["emissivity"],
            absorption_rows["incident_scattering"],
            emissivity_rows["incident_scattering"],
            forward_transmission, reverse_transmission,
        )).astype(float, copy=False)
    except (TypeError, ValueError):
        return None

    if (
        not np.all(np.isfinite(bounded_power))
        or np.any(bounded_power < -POWER_ATOL)
        or np.any(bounded_power > 1.0 + POWER_ATOL)
        or np.any(np.abs(
            forward["R"] + forward["T"] + forward["A"] - 1.0
        ) > POWER_ATOL)
        or np.any(np.abs(
            reverse["R"] + reverse["T"] + reverse["A"] - 1.0
        ) > POWER_ATOL)
        or np.any(np.abs(
            forward_absorption
            - (forward["A"] + forward["T"] - forward_transmission)
        ) > POWER_ATOL)
        or np.any(np.abs(
            reverse_absorption
            - (reverse["A"] + reverse["T"] - reverse_transmission)
        ) > POWER_ATOL)
    ):
        return None

    phi_values = np.column_stack((
        forward["phi_deg"],
        reverse["phi_deg"],
        absorption_rows["phi_deg"],
        emissivity_rows["phi_deg"],
    )).astype(float, copy=False)
    if (
        not np.all(np.isfinite(phi_values))
        or np.any(np.abs(phi_values - phi_values[:, :1]) > ANGLE_ATOL_DEG)
        or np.any(forward["harmonics"] != reverse["harmonics"])
        or np.any(absorption_rows["harmonics"] != forward["harmonics"])
        or np.any(emissivity_rows["harmonics"] != reverse["harmonics"])
    ):
        return None

    out = np.empty(row_count, dtype=FORWARD_REVERSE_DTYPE)
    out["lambda_um"] = forward["lambda_um"]
    out["pol"] = forward["pol"]
    out["theta_deg"] = forward["theta_deg"]
    out["phi_deg"] = forward["phi_deg"]
    out["harmonics"] = forward["harmonics"]
    out["R"] = forward["R"]
    out["T"] = forward_transmission
    out["A"] = forward_absorption
    out["R_reverse"] = reverse["R"]
    out["T_reverse"] = reverse_transmission
    out["A_reverse"] = reverse_absorption
    out["emissivity_same_channel"] = emissivity_rows["emissivity"]
    out["emissivity_reverse_channel"] = absorption_rows["emissivity"]
    out["eta"] = np.abs(
        forward_absorption - emissivity_rows["emissivity"]
    )
    out["conservation"] = (
        forward["R"] + forward_transmission + forward_absorption
    )
    out["reverse_conservation"] = (
        reverse["R"] + reverse_transmission + reverse_absorption
    )
    return out


def combine_forward_reverse_spectrum(
    raw: np.ndarray,
    forward_channels: np.ndarray,
    reverse_channels: np.ndarray,
    *,
    forward_angle_deg: float,
    reverse_angle_deg: float,
) -> np.ndarray:
    def require_fields(rows: np.ndarray, fields: set[str], label: str) -> None:
        if rows.ndim != 1 or rows.dtype.names is None:
            raise ValueError(f"{label} must be a one-dimensional structured array")
        missing = fields.difference(rows.dtype.names)
        if missing:
            raise ValueError(f"{label} is missing fields: {sorted(missing)}")

    raw = np.asarray(raw)
    forward_channels = np.asarray(forward_channels)
    reverse_channels = np.asarray(reverse_channels)
    require_fields(
        raw,
        {
            "lambda_um", "pol", "theta_deg", "phi_deg", "harmonics",
            "R", "T", "A",
        },
        "raw spectrum",
    )
    thermal_fields = {
        "lambda_um", "pol", "theta_deg", "phi_deg", "harmonics",
        "absorptivity", "emissivity", "incident_scattering",
    }
    require_fields(forward_channels, thermal_fields, "forward thermal channels")
    require_fields(reverse_channels, thermal_fields, "reverse thermal channels")

    fast_result = _try_combine_forward_reverse_spectrum_fast(
        raw,
        forward_channels,
        reverse_channels,
        forward_angle_deg,
        reverse_angle_deg,
    )
    if fast_result is not None:
        return fast_result

    def channel_map(
        rows: np.ndarray,
        side: str,
        expected_angle_deg: float,
    ) -> dict[tuple[float, str], np.void]:
        mapped: dict[tuple[float, str], np.void] = {}
        for row in rows:
            angle_deg = float(row["theta_deg"])
            if not np.isclose(
                angle_deg,
                expected_angle_deg,
                rtol=0.0,
                atol=ANGLE_ATOL_DEG,
            ):
                raise ValueError(
                    f"unexpected {side} thermal-channel angle {angle_deg:.12g} deg"
                )
            key = (float(row["lambda_um"]), str(row["pol"]))
            if key in mapped:
                raise ValueError(
                    f"duplicate {side} thermal-channel row for "
                    f"lambda={key[0]:.12g} um, pol={key[1]}"
                )
            mapped[key] = row
        return mapped

    absorption_by_channel = channel_map(
        forward_channels,
        "forward",
        forward_angle_deg,
    )
    emissivity_by_channel = channel_map(
        reverse_channels,
        "reverse",
        reverse_angle_deg,
    )

    def checked_power_row(
        channel_row: np.void,
        spectrum_row: np.void,
        side: str,
    ) -> tuple[float, float]:
        values = np.asarray(
            [
                spectrum_row["R"],
                spectrum_row["T"],
                spectrum_row["A"],
                channel_row["absorptivity"],
                channel_row["emissivity"],
                channel_row["incident_scattering"],
            ],
            dtype=float,
        )
        if not np.all(np.isfinite(values)):
            raise ValueError(f"{side} spectrum/thermal row contains non-finite power")
        reflection, transmission, raw_absorption, absorption, emissivity, incident = values
        for name, value in (
            ("R", reflection),
            ("T", transmission),
            ("A", raw_absorption),
            ("absorptivity", absorption),
            ("emissivity", emissivity),
            ("incident_scattering", incident),
        ):
            if value < -POWER_ATOL or value > 1.0 + POWER_ATOL:
                raise ValueError(f"{side} {name} lies outside [0,1]: {value:.12g}")
        raw_conservation = reflection + transmission + raw_absorption
        if abs(raw_conservation - 1.0) > POWER_ATOL:
            raise ValueError(
                f"{side} raw energy balance failed: "
                f"R+T+A={raw_conservation:.12g}"
            )
        channel_transmission = incident - reflection
        if (
            channel_transmission < -POWER_ATOL
            or channel_transmission > 1.0 + POWER_ATOL
        ):
            raise ValueError(
                f"{side} radiative transmission lies outside [0,1]: "
                f"{channel_transmission:.12g}"
            )
        # Power entering a lossy exterior half-space is reported as T by the
        # raw spectrum, but it is part of the thermodynamic absorptivity: that
        # half-space is not an incoming/outgoing radiative scattering port.
        expected_absorption = (
            raw_absorption + transmission - channel_transmission
        )
        if abs(absorption - expected_absorption) > POWER_ATOL:
            raise ValueError(
                f"{side} spectrum/thermal power partition mismatch: "
                f"absorptivity={absorption:.12g}, "
                f"expected={expected_absorption:.12g}"
            )
        conservation = reflection + channel_transmission + absorption
        if abs(conservation - 1.0) > POWER_ATOL:
            raise ValueError(
                f"{side} energy balance failed: R+T+A={conservation:.12g}"
            )
        return channel_transmission, absorption

    def angle_side(angle_deg: float) -> str:
        if np.isclose(
            angle_deg,
            forward_angle_deg,
            rtol=0.0,
            atol=ANGLE_ATOL_DEG,
        ):
            return "forward"
        if np.isclose(
            angle_deg,
            reverse_angle_deg,
            rtol=0.0,
            atol=ANGLE_ATOL_DEG,
        ):
            return "reverse"
        raise ValueError(
            f"unexpected spectrum angle {angle_deg:.12g} deg; expected "
            f"{forward_angle_deg:.12g} or {reverse_angle_deg:.12g} deg"
        )

    grouped: dict[tuple[float, str], dict[str, np.void]] = {}
    for row in raw:
        key = (float(row["lambda_um"]), str(row["pol"]))
        side = angle_side(float(row["theta_deg"]))
        pair = grouped.setdefault(key, {})
        if side in pair:
            raise ValueError(
                f"duplicate {side} spectrum row for lambda={key[0]:.12g} um, pol={key[1]}"
            )
        pair[side] = row

    spectrum_keys = set(grouped)
    if set(absorption_by_channel) != spectrum_keys:
        raise ValueError("forward thermal-channel keys do not match spectrum keys")
    if set(emissivity_by_channel) != spectrum_keys:
        raise ValueError("reverse thermal-channel keys do not match spectrum keys")

    out = np.empty(len(grouped), dtype=FORWARD_REVERSE_DTYPE)
    for i, ((wavelength_um, polarization), pair) in enumerate(sorted(grouped.items())):
        if "forward" not in pair or "reverse" not in pair:
            missing = "forward" if "forward" not in pair else "reverse"
            raise ValueError(
                f"missing {missing} spectrum row for lambda={wavelength_um:.12g} um, "
                f"pol={polarization}"
            )
        forward = pair["forward"]
        reverse = pair["reverse"]
        channel_key = (wavelength_um, polarization)
        absorption_row = absorption_by_channel.get(channel_key)
        emissivity_row = emissivity_by_channel.get(channel_key)
        if absorption_row is None or emissivity_row is None:
            raise ValueError(
                f"missing directional thermal-channel row for "
                f"lambda={wavelength_um:.12g} um, "
                f"pol={polarization}"
            )
        emissivity = float(emissivity_row["emissivity"])
        transmission, absorption = checked_power_row(
            absorption_row,
            forward,
            "forward",
        )
        reverse_emissivity = float(absorption_row["emissivity"])
        reverse_transmission, reverse_absorption = checked_power_row(
            emissivity_row,
            reverse,
            "reverse",
        )
        phi_values = np.asarray(
            [
                forward["phi_deg"],
                reverse["phi_deg"],
                absorption_row["phi_deg"],
                emissivity_row["phi_deg"],
            ],
            dtype=float,
        )
        if not np.all(np.isfinite(phi_values)) or not np.allclose(
            phi_values,
            phi_values[0],
            rtol=0.0,
            atol=ANGLE_ATOL_DEG,
        ):
            raise ValueError("spectrum and thermal channels use inconsistent phi angles")
        if int(forward["harmonics"]) != int(reverse["harmonics"]):
            raise ValueError("forward and reverse spectra use different harmonic counts")
        if int(absorption_row["harmonics"]) != int(forward["harmonics"]):
            raise ValueError("forward spectrum and thermal channel use different harmonics")
        if int(emissivity_row["harmonics"]) != int(reverse["harmonics"]):
            raise ValueError("reverse spectrum and thermal channel use different harmonics")
        out[i] = (
            float(forward["lambda_um"]),
            polarization,
            float(forward["theta_deg"]),
            float(forward["phi_deg"]),
            int(forward["harmonics"]),
            float(forward["R"]),
            transmission,
            absorption,
            float(reverse["R"]),
            reverse_transmission,
            reverse_absorption,
            emissivity,
            reverse_emissivity,
            abs(absorption - emissivity),
            float(forward["R"]) + transmission + absorption,
            float(reverse["R"]) + reverse_transmission + reverse_absorption,
        )
    return out


def summarize_forward_reverse_spectrum(
    spectrum: np.ndarray,
    polarizations: Sequence[str],
    *,
    include_peak_values: bool = False,
) -> None:
    for polarization in polarizations:
        rows = spectrum[spectrum["pol"] == polarization]
        if len(rows) == 0:
            print(f"{polarization}: no rows")
            continue
        worst = rows[int(np.argmax(rows["eta"]))]
        message = (
            f"{polarization}: max |A-e|={float(worst['eta']):.6f} "
            f"at lambda={float(worst['lambda_um']):.6f} um"
        )
        if include_peak_values:
            message += (
                f"; max A/e="
                f"{float(np.nanmax(rows['A'])):.3f}/"
                f"{float(np.nanmax(rows['emissivity_same_channel'])):.3f}"
            )
        print(message)


def plot_forward_reverse_axes(
    axes: np.ndarray,
    spectrum: np.ndarray,
    polarizations: Sequence[str],
    *,
    markers: dict[str, Sequence[float]] | None = None,
) -> None:
    colors = {"A": "#1f77b4", "emissivity": "#d62728", "eta": "#2ca02c"}
    marker_map = markers or {}
    for ax, polarization in zip(axes, polarizations):
        rows = spectrum[spectrum["pol"] == polarization]
        ax.plot(rows["lambda_um"], rows["A"], color=colors["A"], linewidth=2.2, label="A(+theta)")
        ax.plot(
            rows["lambda_um"],
            rows["emissivity_same_channel"],
            color=colors["emissivity"],
            linewidth=1.9,
            linestyle="--",
            label="e(+theta), scattering-row deficit",
        )
        ax.plot(rows["lambda_um"], rows["eta"], color=colors["eta"], linewidth=1.8, linestyle="-.", label="|A-e|")
        for wavelength_um in marker_map.get(polarization, ()):
            ax.axvline(wavelength_um, color="#777777", linewidth=0.8, alpha=0.28)
        ax.set_title(f"{polarization} polarization")
        ax.set_xlabel("Wavelength (um)")
        ax.set_ylim(-0.03, 1.03)
        ax.grid(True, color="#dddddd", linewidth=0.8, alpha=0.8)
    axes[0].set_ylabel("Spectrum")
    axes[-1].legend(loc="lower right", fontsize=8, frameon=False)


solve_fr = solve_forward_reverse_spectrum
summarize_fr = summarize_forward_reverse_spectrum
plot_fr_axes = plot_forward_reverse_axes
