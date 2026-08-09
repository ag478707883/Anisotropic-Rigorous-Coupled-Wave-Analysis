"""Reproduce Mutlu et al., Optics Letters 37 (2012) 2094, Figs. 3-4.

The binary high-contrast grating uses the backend's analytic one-dimensional
lamellar Fourier coefficients and Li inverse factorization. The plotted
quantities are the zero-order Jones transmission amplitudes, retardance,
circular conversion amplitudes, and conversion efficiency.
"""

from __future__ import annotations

from pathlib import Path

project_root = Path(__file__).resolve().parents[1]

import matplotlib
import numpy as np

import asyrcwa


# Mutlu, Akosman, and Ozbay, Fig. 1 and the paragraph below Eqs. (5)-(6).
period_um = 0.380
ridge_width_um = 0.160
groove_width_um = 0.220
grating_height_um = 0.550
ridge_fill_factor = ridge_width_um / period_um
silicon_refractive_index = 3.48
sio2_refractive_index = 1.47

theta_deg = 0.0
phi_deg = 0.0
polarizations = ("TE", "TM")

# The paper does not state its RCWA order. The grating is one-dimensional, so
# only x orders are retained. Against a (50, 0) audit, (35, 0) keeps the
# full-band complex-transmission error below 1e-5 and the Ceff error below 3e-6.
orders = (35, 0)
workers = 20
# Fourier convergence controls. "Default" preserves the original example result.

wavelength_min_um = 1.0
wavelength_max_um = 3.5
wavelength_point_count = 2501
wavelengths_um = np.linspace(
    wavelength_min_um,
    wavelength_max_um,
    wavelength_point_count,
)

paper_design_wavelength_um = 1.55
paper_two_mode_tm_magnitude = 0.988
paper_two_mode_te_magnitude = 0.936
paper_two_mode_phase_difference_deg = 92.0
paper_rcwa_efficiency_threshold = 0.90
paper_rcwa_band_um = (1.36, 2.36)
paper_rcwa_bandwidth_percent = 54.0
paper_fdtd_band_um = (1.40, 2.36)
paper_fdtd_bandwidth_percent = 51.0

plot_path = (
    project_root
    / "images"
    / "mutlu_2012_hcg_circular_polarizer_spectrum.png"
)
figure_size = (12.0, 8.0)
figure_dpi = 240
show_plot = True

if not show_plot:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt


def build_simulation(rcwa: object) -> object:
    if not np.isclose(
        ridge_width_um + groove_width_um,
        period_um,
        rtol=0.0,
        atol=1e-12,
    ):
        raise ValueError("ridge_width_um + groove_width_um must equal period_um")

    sim = rcwa.New(
        Lattice=period_um,
        Orders=orders,
    )
    sim.SetMaterialModel("Si", "Si")
    sim.SetMaterial("SiO2", sio2_refractive_index**2 + 0.0j)
    sim.SetSuperstrate("SiO2")
    sim.SetSubstrate("SiO2")
    sim.AddLayer("Si_SiO2_HCG", grating_height_um, "SiO2")
    sim.SetRegionRectangle(
        "Si_SiO2_HCG",
        "Si",
        Center=(0.0, 0.0),
        Angle=0.0,
        Halfwidths=(0.5 * ridge_fill_factor * period_um, 0.5 * period_um),
    )
    return sim


def complex_amplitude(
    rows: np.ndarray,
    block: str,
    out_polarization: str,
    in_polarization: str,
) -> np.ndarray:
    selected = rows[
        (rows["block"] == block)
        & (rows["out_pol"] == out_polarization)
        & (rows["in_pol"] == in_polarization)
    ]
    selected = np.sort(selected, order="lambda_um")
    if len(selected) != wavelength_point_count:
        raise RuntimeError(
            f"incomplete {block}_{out_polarization}{in_polarization} amplitude"
        )
    if not np.allclose(
        selected["lambda_um"],
        wavelengths_um,
        rtol=0.0,
        atol=1e-12,
    ):
        raise RuntimeError("zero-order amplitude wavelengths are misaligned")
    return selected["value_re"] + 1j * selected["value_im"]


def threshold_band(
    wavelengths: np.ndarray,
    values: np.ndarray,
    threshold: float,
) -> tuple[float, float]:
    above = values >= threshold
    starts = np.flatnonzero(above & np.r_[True, ~above[:-1]])
    ends = np.flatnonzero(above & np.r_[~above[1:], True])
    if len(starts) == 0:
        raise RuntimeError(f"no conversion-efficiency band exceeds {threshold:g}")

    lengths = ends - starts
    segment = int(np.argmax(lengths))
    start = int(starts[segment])
    end = int(ends[segment])

    def crossing(index_a: int, index_b: int) -> float:
        value_a = float(values[index_a])
        value_b = float(values[index_b])
        fraction = (threshold - value_a) / (value_b - value_a)
        return float(
            wavelengths[index_a]
            + fraction * (wavelengths[index_b] - wavelengths[index_a])
        )

    lower = (
        float(wavelengths[start])
        if start == 0
        else crossing(start - 1, start)
    )
    upper = (
        float(wavelengths[end])
        if end == len(wavelengths) - 1
        else crossing(end, end + 1)
    )
    return lower, upper


def bandwidth_percent(lower_um: float, upper_um: float) -> float:
    return 200.0 * (upper_um / lower_um - 1.0) / (upper_um / lower_um + 1.0)


def solve_spectrum() -> dict[str, np.ndarray | float | tuple[float, float]]:
    simulation = build_simulation(asyrcwa)
    amplitude_rows = simulation.GetZeroOrderAmplitudes(
        wavelengths_um,
        Theta=theta_deg,
        Phi=phi_deg,
        Workers=workers,
    )

    t_te = complex_amplitude(amplitude_rows, "T", "TE", "TE")
    t_tm = complex_amplitude(amplitude_rows, "T", "TM", "TM")
    t_te_tm = complex_amplitude(amplitude_rows, "T", "TE", "TM")
    t_tm_te = complex_amplitude(amplitude_rows, "T", "TM", "TE")
    r_te = complex_amplitude(amplitude_rows, "R", "TE", "TE")
    r_tm = complex_amplitude(amplitude_rows, "R", "TM", "TM")

    # The paper uses C_+ = (T_TM - i*T_TE)/2 and C_- =
    # (T_TM + i*T_TE)/2 for a 45-degree linear input.
    c_rcp = 0.5 * (t_tm - 1j * t_te)
    c_lcp = 0.5 * (t_tm + 1j * t_te)
    rcp_power = np.abs(c_rcp) ** 2
    lcp_power = np.abs(c_lcp) ** 2
    conversion_efficiency = (rcp_power - lcp_power) / (rcp_power + lcp_power)

    # The backend uses exp(-i*omega*t), opposite to the phase convention used
    # for the positive retardance in Fig. 3(b), so the plotted paper-convention
    # value is angle(T_TE/T_TM), wrapped to [0, 360) degrees.
    phase_difference_deg = np.mod(
        np.degrees(np.angle(t_te / t_tm)),
        360.0,
    )
    efficiency_band_um = threshold_band(
        wavelengths_um,
        conversion_efficiency,
        paper_rcwa_efficiency_threshold,
    )

    conservation_error = max(
        float(np.max(abs(np.abs(r_te) ** 2 + np.abs(t_te) ** 2 - 1.0))),
        float(np.max(abs(np.abs(r_tm) ** 2 + np.abs(t_tm) ** 2 - 1.0))),
    )
    cross_polarization_max = max(
        float(np.max(np.abs(t_te_tm))),
        float(np.max(np.abs(t_tm_te))),
    )

    return {
        "t_te": t_te,
        "t_tm": t_tm,
        "c_rcp": c_rcp,
        "c_lcp": c_lcp,
        "phase_difference_deg": phase_difference_deg,
        "conversion_efficiency": conversion_efficiency,
        "efficiency_band_um": efficiency_band_um,
        "conservation_error": conservation_error,
        "cross_polarization_max": cross_polarization_max,
        "harmonics": float(amplitude_rows["harmonics"][0]),
    }


def print_paper_comparison(
    spectrum: dict[str, np.ndarray | float | tuple[float, float]],
) -> None:
    index = int(np.argmin(abs(wavelengths_um - paper_design_wavelength_um)))
    t_tm = np.asarray(spectrum["t_tm"])
    t_te = np.asarray(spectrum["t_te"])
    phase_difference_deg = np.asarray(spectrum["phase_difference_deg"])
    efficiency = np.asarray(spectrum["conversion_efficiency"])
    lower_um, upper_um = spectrum["efficiency_band_um"]
    computed_bandwidth_percent = bandwidth_percent(lower_um, upper_um)

    print(
        f"At {paper_design_wavelength_um:.2f} um: "
        f"|T_TM|/|T_TE|={abs(t_tm[index]):.6f}/{abs(t_te[index]):.6f}, "
        f"phase difference={phase_difference_deg[index]:.3f} deg, "
        f"Ceff={efficiency[index]:.6f}"
    )
    print(
        "Paper two-mode values: "
        f"|T_TM|/|T_TE|={paper_two_mode_tm_magnitude:.3f}/"
        f"{paper_two_mode_te_magnitude:.3f}, "
        f"phase difference={paper_two_mode_phase_difference_deg:.0f} deg"
    )
    print(
        f"Computed Ceff >= {paper_rcwa_efficiency_threshold:.2f}: "
        f"{lower_um:.4f}-{upper_um:.4f} um, "
        f"BW={computed_bandwidth_percent:.2f}%; paper RCWA "
        f"{paper_rcwa_band_um[0]:.2f}-{paper_rcwa_band_um[1]:.2f} um, "
        f"BW={paper_rcwa_bandwidth_percent:.0f}%"
    )
    print(
        "Paper FDTD/Palik reference (not recalculated here): "
        f"{paper_fdtd_band_um[0]:.2f}-{paper_fdtd_band_um[1]:.2f} um, "
        f"BW={paper_fdtd_bandwidth_percent:.0f}%"
    )
    print(
        f"retained harmonics={int(spectrum['harmonics'])}, "
        f"max power-conservation error={spectrum['conservation_error']:.3e}, "
        f"max cross-polarized T amplitude={spectrum['cross_polarization_max']:.3e}"
    )


def plot_spectrum(
    spectrum: dict[str, np.ndarray | float | tuple[float, float]],
) -> None:
    t_te = np.asarray(spectrum["t_te"])
    t_tm = np.asarray(spectrum["t_tm"])
    c_rcp = np.asarray(spectrum["c_rcp"])
    c_lcp = np.asarray(spectrum["c_lcp"])
    phase_difference_deg = np.asarray(spectrum["phase_difference_deg"])
    efficiency = np.asarray(spectrum["conversion_efficiency"])
    lower_um, upper_um = spectrum["efficiency_band_um"]

    fig, axes = plt.subplots(
        2,
        2,
        figsize=figure_size,
        constrained_layout=True,
    )
    axes[0, 0].plot(wavelengths_um, abs(t_tm), color="black", label="TM")
    axes[0, 0].plot(
        wavelengths_um,
        abs(t_te),
        color="#e52d2d",
        linestyle="--",
        label="TE",
    )
    axes[0, 0].set_ylabel("Field transmission")
    axes[0, 0].set_ylim(0.78, 1.01)
    axes[0, 0].set_title("Fig. 3(a): zero-order transmission amplitudes")
    axes[0, 0].legend()

    axes[0, 1].plot(
        wavelengths_um,
        phase_difference_deg,
        color="#264fba",
        linestyle="-.",
    )
    axes[0, 1].set_ylabel("Phase difference (deg)")
    axes[0, 1].set_ylim(30.0, 140.0)
    axes[0, 1].set_title("Fig. 3(b): phase retardance")

    axes[1, 0].plot(wavelengths_um, abs(c_rcp), color="black", label="RCP (+)")
    axes[1, 0].plot(
        wavelengths_um,
        abs(c_lcp),
        color="#e52d2d",
        linestyle="--",
        label="LCP (-)",
    )
    axes[1, 0].set_ylabel("Circular conversion amplitude")
    axes[1, 0].set_ylim(-0.02, 1.02)
    axes[1, 0].set_title("Fig. 4(a): circular conversion coefficients")
    axes[1, 0].legend()

    axes[1, 1].plot(
        wavelengths_um,
        efficiency,
        color="#264fba",
        label="RCWA",
    )
    axes[1, 1].axhline(
        paper_rcwa_efficiency_threshold,
        color="0.4",
        linestyle=":",
        linewidth=1.0,
        label="0.9 threshold",
    )
    axes[1, 1].axvspan(
        lower_um,
        upper_um,
        color="#264fba",
        alpha=0.08,
        label="computed band",
    )
    for paper_edge_um in paper_rcwa_band_um:
        axes[1, 1].axvline(
            paper_edge_um,
            color="#e52d2d",
            linestyle="--",
            linewidth=0.9,
            alpha=0.65,
        )
    axes[1, 1].set_ylabel("Conversion efficiency")
    axes[1, 1].set_ylim(0.55, 1.01)
    axes[1, 1].set_title("Fig. 4(b): circular conversion efficiency")
    axes[1, 1].legend(loc="lower left")

    for ax in axes.flat:
        ax.set_xlim(wavelength_min_um, wavelength_max_um)
        ax.set_xlabel("Wavelength (um)")
        ax.axvline(
            paper_design_wavelength_um,
            color="0.65",
            linewidth=0.7,
            alpha=0.6,
        )

    fig.suptitle(
        "Mutlu et al. (2012) Si/SiO2 high-contrast-grating circular polarizer"
    )
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(plot_path, dpi=figure_dpi)
    print(f"Saved {plot_path}.")
    plt.show() if show_plot else plt.close(fig)


def main() -> dict[str, np.ndarray | float | tuple[float, float]]:
    print(f"Using asyrcwa backend from {asyrcwa.module_directory()}.")
    print(
        f"period={period_um * 1000:.0f} nm, ridge/groove="
        f"{ridge_width_um * 1000:.0f}/{groove_width_um * 1000:.0f} nm, "
        f"height={grating_height_um * 1000:.0f} nm, "
        f"polarizations={'/'.join(polarizations)}, orders={orders}"
    )
    with asyrcwa.timed_step("solve Mutlu et al. Figs. 3-4 Jones spectrum"):
        spectrum = solve_spectrum()
        print_paper_comparison(spectrum)
    with asyrcwa.timed_step("plot spectrum"):
        plot_spectrum(spectrum)
    return spectrum


if __name__ == "__main__":
    main()
