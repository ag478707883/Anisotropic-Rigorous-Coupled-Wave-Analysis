"""Reproduce Fang et al., ICHMT 159 (2024) 108147, Fig. 2(c,d).

The silicon square rings use analytic nested-rectangle Fourier coefficients.
The local RCWA backend computes the TE absorptivity, emissivity, and
nonreciprocity at the paper's +/-4-degree incidence pair.
"""

from __future__ import annotations

from pathlib import Path

project_root = Path(__file__).resolve().parents[1]

import matplotlib
import numpy as np

import asyrcwa
from asyrcwa.spectrum import solve_fr, summarize_fr


# Fang et al., Fig. 1, Eq. (1), and Section 3.
period_um = 1.20
ring_inner_width_ratio = 0.30
ring_outer_width_ratio = 0.70
ring_inner_width_um = ring_inner_width_ratio * period_um
ring_outer_width_um = ring_outer_width_ratio * period_um
ring_height_um = 0.50
magneto_optical_height_um = 0.65
ag_height_um = 0.20
silicon_refractive_index = 3.48

# The paper assumes epsilon=4 and b=0.1 at a typical 100-Oe saturation
# magnetization. Backend +y gives epsilon_xz=-i*b and epsilon_zx=+i*b,
# matching Eq. (1).
saturation_magnetization_oe = 100.0
ceyig_parameters = {
    "diagonal_epsilon": 4.0,
    "offdiagonal_gyration": 0.10,
    "magnetization_direction": (0.0, 1.0, 0.0),
}

# The article refers to Zhao and Zhang, JQSRT 135 (2014), for Ag without
# printing its optical constants. This is the repository's explicit NIR Ag
# Drude approximation and is therefore a modeling assumption, not a paper fit.
ag_parameters = {
    "eps_inf": 3.4,
    "plasma_frequency_rad_s": 1.39e16,
    "damping_rate_rad_s": 2.7e13,
}

theta_deg = 4.0
phi_deg = 0.0
polarizations = ("TE",)
angles_deg = (+theta_deg, -theta_deg)

# The paper does not report its Fourier order. This practical (6, 6) setting
# retains 113 harmonics. A production (9, 9) audit retains 253 harmonics and
# leaves a 0.3-0.9-nm residual shift relative to (8, 8).
orders = (6, 6)
workers = 14
# Fourier convergence controls. "Default" preserves the original example result.
workers_per_angle = max(1, workers // len(angles_deg))

paper_wavelength_min_um = 1.650
paper_wavelength_max_um = 1.780
# The computed fourth resonance converges just above the plotted paper range,
# so the solve is extended by 10 nm rather than clipping that peak.
wavelength_min_um = paper_wavelength_min_um
wavelength_max_um = 1.790
wavelength_point_count = 1401
wavelengths_um = np.linspace(
    wavelength_min_um,
    wavelength_max_um,
    wavelength_point_count,
)

paper_absorption_peaks_um = (1.6688, 1.6943, 1.7280, 1.7762)
paper_emission_peaks_um = (1.6593, 1.6922, 1.7405, 1.7778)
paper_nonreciprocity_lower_bound = 0.70

plot_path = (
    project_root
    / "images"
    / "fang_2024_te_multiband_si_square_rings_fig2.png"
)
figure_size = (9.5, 7.2)
figure_dpi = 240
show_plot = True

if not show_plot:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt


def build_simulation(rcwa: object) -> object:
    if not 0.0 < ring_inner_width_um < ring_outer_width_um < period_um:
        raise ValueError(
            "ring widths must satisfy 0 < inner width < outer width < period"
        )

    sim = rcwa.New(
        Lattice=(period_um, period_um),
        Orders=orders,
    )
    sim.SetMaterialModel("Air", "Air")
    sim.SetMaterialModel("Si", "Si")
    sim.SetMaterialModel("CeYIG", "YIG", ceyig_parameters)
    sim.SetMaterialModel("Ag", "Drude", ag_parameters)
    sim.SetSuperstrate("Air")
    sim.SetSubstrate("Si")

    # Air background + outer Si square + inner Air square is one isolated
    # square ring represented by exact nested-rectangle Fourier coefficients.
    sim.AddLayer("square_rings", ring_height_um, "Air")
    sim.SetRegionRectangle(
        "square_rings",
        "Si",
        Center=(0.0, 0.0),
        Angle=0.0,
        Halfwidths=(0.5 * ring_outer_width_um, 0.5 * ring_outer_width_um),
    )
    sim.SetRegionRectangle(
        "square_rings",
        "Air",
        Center=(0.0, 0.0),
        Angle=0.0,
        Halfwidths=(0.5 * ring_inner_width_um, 0.5 * ring_inner_width_um),
    )
    sim.AddLayer("CeYIG", magneto_optical_height_um, "CeYIG")
    sim.AddLayer("Ag", ag_height_um, "Ag")
    return sim


def strongest_local_peaks(
    rows: np.ndarray,
    field: str,
    count: int,
) -> np.ndarray:
    values = np.asarray(rows[field], dtype=float)
    indices = np.flatnonzero(
        (values[1:-1] > values[:-2]) & (values[1:-1] >= values[2:])
    ) + 1
    if len(indices) < count:
        raise RuntimeError(f"found only {len(indices)} local peaks in {field}")
    ranked = indices[np.argsort(values[indices])[::-1]]
    selected = ranked[:count]
    return selected[np.argsort(rows["lambda_um"][selected])]


def print_peak_comparison(spectrum: np.ndarray) -> None:
    rows = spectrum[spectrum["pol"] == "TE"]
    computed_absorption = strongest_local_peaks(rows, "A", 4)
    computed_emission = strongest_local_peaks(
        rows,
        "emissivity_same_channel",
        4,
    )

    print("TE absorption peaks (computed versus paper):")
    for index, paper_wavelength_um in zip(
        computed_absorption,
        paper_absorption_peaks_um,
    ):
        computed_wavelength_um = float(rows["lambda_um"][index])
        print(
            f"  {computed_wavelength_um * 1000:.2f} nm, "
            f"A/e/eta={rows['A'][index]:.6f}/"
            f"{rows['emissivity_same_channel'][index]:.6f}/"
            f"{rows['eta'][index]:.6f}; paper "
            f"{paper_wavelength_um * 1000:.1f} nm, "
            f"delta={(computed_wavelength_um - paper_wavelength_um) * 1000:+.2f} nm"
        )

    print("TE emission peaks (computed versus paper):")
    for index, paper_wavelength_um in zip(
        computed_emission,
        paper_emission_peaks_um,
    ):
        computed_wavelength_um = float(rows["lambda_um"][index])
        print(
            f"  {computed_wavelength_um * 1000:.2f} nm, "
            f"A/e/eta={rows['A'][index]:.6f}/"
            f"{rows['emissivity_same_channel'][index]:.6f}/"
            f"{rows['eta'][index]:.6f}; paper "
            f"{paper_wavelength_um * 1000:.1f} nm, "
            f"delta={(computed_wavelength_um - paper_wavelength_um) * 1000:+.2f} nm"
        )

    max_transmission = max(
        float(np.max(rows["T"])),
        float(np.max(rows["T_reverse"])),
    )
    paper_emissivity_gap = float(np.max(abs(
        1.0 - rows["R_reverse"] - rows["emissivity_same_channel"]
    )))

    band_eta = []
    for absorption_index, emission_index in zip(
        computed_absorption,
        computed_emission,
    ):
        lower_index = min(int(absorption_index), int(emission_index))
        upper_index = max(int(absorption_index), int(emission_index))
        band_eta.append(float(np.max(rows["eta"][lower_index : upper_index + 1])))
    print(
        f"max eta={float(np.max(rows['eta'])):.6f} "
        f"(paper: >{paper_nonreciprocity_lower_bound:.2f}); "
        f"band maxima={tuple(round(value, 4) for value in band_eta)}; "
        f"max T={max_transmission:.3e}; "
        f"max |1-R(-theta)-exact e|={paper_emissivity_gap:.3e}"
    )


def plot_spectrum(spectrum: np.ndarray) -> None:
    rows = spectrum[spectrum["pol"] == "TE"]
    wavelength_nm = 1000.0 * rows["lambda_um"]

    fig, axes = plt.subplots(
        2,
        1,
        figsize=figure_size,
        sharex=True,
        constrained_layout=True,
    )
    axes[0].plot(
        wavelength_nm,
        rows["emissivity_same_channel"],
        color="#2455d6",
        linewidth=1.8,
        label=r"$e$",
    )
    axes[0].plot(
        wavelength_nm,
        rows["A"],
        color="#e02f2f",
        linewidth=1.8,
        label=r"$\alpha$",
    )
    axes[0].set_ylabel(r"$\alpha$ and $e$")
    axes[0].set_ylim(-0.02, 1.03)
    axes[0].legend(loc="upper right")
    axes[0].set_title("(c) TE absorptivity and emissivity")

    axes[1].plot(
        wavelength_nm,
        rows["eta"],
        color="#e02f2f",
        linewidth=1.8,
        label=r"$\eta=|\alpha-e|$",
    )
    axes[1].axhline(
        paper_nonreciprocity_lower_bound,
        color="0.45",
        linewidth=1.0,
        linestyle=":",
        label="paper lower bound",
    )
    axes[1].set_xlabel("Wavelength (nm)")
    axes[1].set_ylabel(r"$\eta$")
    axes[1].set_ylim(-0.02, 1.03)
    axes[1].set_xlim(1000.0 * wavelength_min_um, 1000.0 * wavelength_max_um)
    axes[1].legend(loc="upper right")
    axes[1].set_title("(d) TE nonreciprocity")

    for wavelength_um in paper_absorption_peaks_um:
        axes[0].axvline(
            1000.0 * wavelength_um,
            color="#e02f2f",
            linewidth=0.7,
            alpha=0.30,
        )
    for wavelength_um in paper_emission_peaks_um:
        axes[0].axvline(
            1000.0 * wavelength_um,
            color="#2455d6",
            linewidth=0.7,
            alpha=0.30,
        )

    fig.suptitle(
        "Fang et al. (2024): Si square rings / Ce:YIG / Ag / Si, "
        f"theta=+/-{theta_deg:g} deg, orders={orders}"
    )
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(plot_path, dpi=figure_dpi)
    print(f"Saved {plot_path}.")
    plt.show() if show_plot else plt.close(fig)


def main() -> np.ndarray:
    print(f"Using asyrcwa backend from {asyrcwa.module_directory()}.")
    print(
        f"period={period_um:g} um, inner/outer widths="
        f"{ring_inner_width_um:g}/{ring_outer_width_um:g} um, "
        f"n(Si)={silicon_refractive_index:g}, "
        f"saturation magnetization={saturation_magnetization_oe:g} Oe"
    )
    with asyrcwa.timed_step("solve Fang et al. Fig. 2(c,d) TE spectrum"):
        spectrum = solve_fr(
            rcwa=asyrcwa,
            build_simulation=build_simulation,
            wavelengths_um=wavelengths_um,
            polarizations=polarizations,
            angles_deg=angles_deg,
            phi_deg=phi_deg,
            workers_per_angle=workers_per_angle,
        )
        summarize_fr(spectrum, polarizations, include_peak_values=True)
        print_peak_comparison(spectrum)
    with asyrcwa.timed_step("plot spectrum"):
        plot_spectrum(spectrum)
    return spectrum


if __name__ == "__main__":
    main()
