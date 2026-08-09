"""Reproduce Fang et al., IJHMT 223 (2024) 125229, Fig. 2(c,d).

The paper uses square silicon annular arrays on a Ce:YIG magneto-optical film
and an Ag reflector.  This example computes the directional absorptivity,
emissivity, and nonreciprocity with the local RCWA backend.
"""

from __future__ import annotations

from pathlib import Path

project_root = Path(__file__).resolve().parents[1]

import matplotlib
import numpy as np

import asyrcwa
from asyrcwa.spectrum import plot_fr_axes, solve_fr, summarize_fr


# Fang et al., Table 1 and Fig. 1.
period_um = 1.30
annulus_outer_width_ratio = 0.70
annulus_inner_width_ratio = 0.30
annulus_outer_width_um = annulus_outer_width_ratio * period_um
annulus_inner_width_um = annulus_inner_width_ratio * period_um
annulus_height_um = 0.60
ceyig_height_um = 0.80
ag_height_um = 0.20
silicon_refractive_index = 3.48

# The paper assumes lossless Ce:YIG with epsilon=4 and b=0.1.  A +y
# magnetization gives epsilon_xz=-i*b and epsilon_zx=+i*b in the backend's
# exp(-i omega t) convention, matching Eq. (1) of the paper.
ceyig_parameters = {
    "diagonal_epsilon": 4.0,
    "offdiagonal_gyration": 0.10,
    "magnetization_direction": (0.0, 1.0, 0.0),
}
ag_parameters = {
    "eps_inf": 3.4,
    "plasma_frequency_rad_s": 1.39e16,
    "damping_rate_rad_s": 2.7e13,
}

theta_deg = 0.8
phi_deg = 0.0
polarizations = ("TE", "TM")
angles_deg = (+theta_deg, -theta_deg)

# The article does not state its retained Fourier order.  This explicit
# circular truncation resolves the square-ring interfaces while keeping the
# example practical to rerun; a denser order can be used for Q-factor checks.
orders = (6, 6)
workers = 20
# Fourier convergence controls. "Default" preserves the original example result.
workers_per_angle = max(1, workers // len(angles_deg))

# The reported resonance wavelengths are 1684.90 nm (TE) and 1669.74 nm
# (TM), with Q approximately 4825.  Use a sub-0.1-nm grid around both peaks.
wavelength_min_um = 1.66
wavelength_max_um = 1.695
wavelength_point_count = 701
wavelengths_um = np.linspace(
    wavelength_min_um,
    wavelength_max_um,
    wavelength_point_count,
)

paper_absorption_peak_um = {"TE": 1.68550, "TM": 1.66924}
paper_emission_peak_um = {"TE": 1.68490, "TM": 1.66974}
paper_peak_eta_lower_bound = 0.85
paper_markers = {
    polarization: (
        paper_absorption_peak_um[polarization],
        paper_emission_peak_um[polarization],
    )
    for polarization in polarizations
}

plot_path = (
    project_root
    / "images"
    / "fang_2024_dual_polarization_annular_arrays_fig2.png"
)
figure_size = (12.0, 4.8)
figure_dpi = 240
show_plot = True  # Set to False to save the figure without displaying it.

if not show_plot:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt


def build_simulation(rcwa: object) -> object:
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

    # Background Air + outer Si square + inner Air square gives the analytic
    # nested-rectangle Fourier coefficients of one square silicon annulus.
    sim.AddLayer("Si_square_annulus", annulus_height_um, "Air")
    sim.SetRegionRectangle(
        "Si_square_annulus",
        "Si",
        Center=(0.0, 0.0),
        Angle=0.0,
        Halfwidths=(0.5 * annulus_outer_width_um, 0.5 * annulus_outer_width_um),
    )
    sim.SetRegionRectangle(
        "Si_square_annulus",
        "Air",
        Center=(0.0, 0.0),
        Angle=0.0,
        Halfwidths=(0.5 * annulus_inner_width_um, 0.5 * annulus_inner_width_um),
    )
    sim.AddLayer("CeYIG", ceyig_height_um, "CeYIG")
    sim.AddLayer("Ag", ag_height_um, "Ag")
    return sim


def nearest_row(rows: np.ndarray, wavelength_um: float) -> np.void:
    return rows[int(np.argmin(abs(rows["lambda_um"] - wavelength_um)))]


def print_paper_comparison(spectrum: np.ndarray) -> None:
    for polarization in polarizations:
        rows = spectrum[spectrum["pol"] == polarization]
        computed_peak = rows[int(np.argmax(rows["eta"]))]
        absorption_reference = nearest_row(
            rows,
            paper_absorption_peak_um[polarization],
        )
        emission_reference = nearest_row(
            rows,
            paper_emission_peak_um[polarization],
        )
        print(
            f"{polarization}: computed max eta={computed_peak['eta']:.6f} at "
            f"{computed_peak['lambda_um'] * 1000:.2f} nm"
        )
        print(
            f"  paper absorption peak {paper_absorption_peak_um[polarization] * 1000:.2f} nm: "
            f"A/e/eta={absorption_reference['A']:.6f}/"
            f"{absorption_reference['emissivity_same_channel']:.6f}/"
            f"{absorption_reference['eta']:.6f}"
        )
        print(
            f"  paper emission peak {paper_emission_peak_um[polarization] * 1000:.2f} nm: "
            f"A/e/eta={emission_reference['A']:.6f}/"
            f"{emission_reference['emissivity_same_channel']:.6f}/"
            f"{emission_reference['eta']:.6f}; "
            f"paper peak eta > {paper_peak_eta_lower_bound:.2f}"
        )


def plot_spectrum(spectrum: np.ndarray) -> None:
    fig, axes = plt.subplots(
        1,
        2,
        figsize=figure_size,
        sharey=True,
        constrained_layout=True,
    )
    plot_fr_axes(axes, spectrum, polarizations, markers=paper_markers)
    for ax in axes:
        ax.set_xlim(wavelength_min_um, wavelength_max_um)
        ax.set_xlabel("Wavelength (um)")
    fig.suptitle(
        "Si square annulus / Ce:YIG / Ag / Si, "
        f"theta=+/-{theta_deg:g} deg, orders={orders}"
    )
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(plot_path, dpi=figure_dpi)
    print(f"Saved {plot_path}.")
    plt.show() if show_plot else plt.close(fig)


def main() -> np.ndarray:
    print(f"Using asyrcwa backend from {asyrcwa.module_directory()}.")
    with asyrcwa.timed_step("solve Fang et al. Fig. 2(c,d) spectrum"):
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
        print_paper_comparison(spectrum)
    with asyrcwa.timed_step("plot spectrum"):
        plot_spectrum(spectrum)
    return spectrum


if __name__ == "__main__":
    main()
