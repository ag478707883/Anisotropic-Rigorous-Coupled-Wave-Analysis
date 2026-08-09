"""Reproduce Shi and Wang, J. Mater. Chem. A 13 (2025) 17578, Fig. 2(a,b)."""

from __future__ import annotations

from pathlib import Path

project_root = Path(__file__).resolve().parents[1]

import matplotlib
import numpy as np

import asyrcwa
from asyrcwa.spectrum import plot_fr_axes, solve_fr, summarize_fr


# Shi and Wang, Fig. 1 and Table 1.
period_um = 7.70
outer_square_width_um = 6.16
circular_hole_diameter_um = 3.08
ge_hollow_array_height_um = 8.13
inas_height_um = 5.44
ag_height_um = 1.00
# The paper does not tabulate dispersive Ge or SiO2 data; these are the
# repository's explicit nondispersive mid-IR assumptions for those two media.
ge_refractive_index = 4.00
sio2_refractive_index = 1.45

# The paper uses the standard magnetized-InAs tensor from refs. 44-46.  These
# constants reproduce wp=2.7396e14 rad/s with m*=0.033 me.
inas_parameters = {
    "magnetic_field_t": +0.8,
    "eps_inf": 12.37,
    "carrier_density_cm3": 7.782250266240598e17,
    "effective_mass_ratio": 0.033,
    "damping_rate_rad_s": 1.55e11,
}
ag_parameters = {
    "eps_inf": 3.4,
    "plasma_frequency_rad_s": 1.39e16,
    "damping_rate_rad_s": 2.7e13,
}

theta_deg = 4.7
phi_deg = 0.0
polarizations = ("TE", "TM")
angles_deg = (+theta_deg, -theta_deg)

# Fig. 10(a) reports that (8,8) spatial harmonics are sufficient and that
# increasing the order to (10,10) produces almost no further change.  The
# paper's pair denotes the retained x/y order limits, so use the full
# parallelogramic (2*8+1)^2 basis rather than circular truncation.
orders = (6, 6)
workers = 20
# Fourier convergence controls. "Default" preserves the original example result.
workers_per_angle = max(1, workers // len(angles_deg))

wavelength_min_um = 17.20
wavelength_max_um = 18.20
wavelength_point_count = 1001
wavelengths_um = np.linspace(
    wavelength_min_um,
    wavelength_max_um,
    wavelength_point_count,
)

# Peak positions stated in Section 3 for Fig. 2(a,b).
paper_emission_peaks_um = {
    "TE": (17.50, 17.89, 18.01),
    "TM": (17.43, 17.57, 17.73, 17.93),
}
paper_absorption_peaks_um = {
    "TE": (17.52, 17.90, 18.00),
    "TM": (17.34, 17.55, 17.77, 18.02),
}
paper_markers = {
    polarization: tuple(
        sorted(
            paper_emission_peaks_um[polarization]
            + paper_absorption_peaks_um[polarization]
        )
    )
    for polarization in polarizations
}

plot_path = project_root / "images" / "shi_2025_ge_hollow_array_inas_ag_fig2.png"
figure_size = (12.0, 4.8)
figure_dpi = 240
show_plot = True

if not show_plot:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt


def build_simulation(rcwa: object) -> object:
    sim = rcwa.New(
        Lattice=(period_um, period_um),
        Orders=orders,
    )
    sim.SetMaterialModel("Air", "Air")
    sim.SetMaterialModel("Ge", "Ge")
    sim.SetMaterialModel("InAs", "InAsMagneto", inas_parameters)
    sim.SetMaterialModel("Ag", "Drude", ag_parameters)
    sim.SetMaterial("SiO2", sio2_refractive_index**2 + 0.0j)
    sim.SetSuperstrate("Air")
    sim.SetSubstrate("SiO2")

    # Every layer starts with AddLayer. SetRegion... makes this layer patterned;
    # its RETICOLO rectangles then use exact rectangular Fourier integrals.
    sim.AddLayer(
        "Ge_square_with_circular_hole",
        ge_hollow_array_height_um,
        "Air",
    )
    sim.SetRegionRectangle(
        "Ge_square_with_circular_hole",
        "Ge",
        Center=(0.0, 0.0),
        Angle=0.0,
        Halfwidths=(0.5 * outer_square_width_um, 0.5 * outer_square_width_um),
    )
    sim.SetRegionCircle(
        "Ge_square_with_circular_hole",
        "Air",
        Center=(0.0, 0.0),
        Radius=0.5 * circular_hole_diameter_um,
    )
    sim.AddLayer("InAs", inas_height_um, "InAs")
    sim.AddLayer("Ag", ag_height_um, "Ag")
    return sim


def nearest_row(rows: np.ndarray, wavelength_um: float) -> np.void:
    return rows[int(np.argmin(abs(rows["lambda_um"] - wavelength_um)))]


def print_paper_comparison(spectrum: np.ndarray) -> None:
    for polarization in polarizations:
        rows = spectrum[spectrum["pol"] == polarization]
        print(f"{polarization} paper emission-channel checkpoints:")
        for wavelength_um in paper_emission_peaks_um[polarization]:
            row = nearest_row(rows, wavelength_um)
            print(
                f"  {wavelength_um:.2f} um: A/e/eta="
                f"{row['A']:.4f}/{row['emissivity_same_channel']:.4f}/"
                f"{row['eta']:.4f}"
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
    fig.suptitle(
        "Ge square/circular-hole array / InAs / Ag / SiO2, "
        f"B=0.8 T, theta=+/-{theta_deg:g} deg, orders={orders}"
    )
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(plot_path, dpi=figure_dpi)
    print(f"Saved {plot_path}.")
    plt.show() if show_plot else plt.close(fig)


def main() -> np.ndarray:
    print(f"Using asyrcwa backend from {asyrcwa.module_directory()}.")
    with asyrcwa.timed_step("solve Shi and Wang Fig. 2 spectrum"):
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
