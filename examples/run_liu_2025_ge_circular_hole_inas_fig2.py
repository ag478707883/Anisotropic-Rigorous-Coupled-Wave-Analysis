"""Recompute Liu et al., Physica Scripta 100 (2025) 115527, Fig. 2."""

from __future__ import annotations

from pathlib import Path

project_root = Path(__file__).resolve().parents[1]

import matplotlib
import numpy as np

import asyrcwa
from asyrcwa.spectrum import plot_fr_axes, solve_fr, summarize_fr


# Liu, Wang, and Xiao, Fig. 1 and Table 1. The table repeats h2; its 6.27 um
# entry is h1 (Ge), while 4.50 um is h2 (InAs), as defined in the text.
period_um = 8.30
ge_square_width_um = 4.08
circular_hole_diameter_um = 0.25
ge_height_um = 6.27
inas_height_um = 4.50
al_height_um = 0.20

# The paper omits Ge and SiO2 optical constants. Ge uses the repository's
# fixed mid-IR model (n=4.0); n=1.45 is used for the SiO2 substrate.
sio2_refractive_index = 1.45
inas_parameters = {
    "magnetic_field_t": +2.0,
    "eps_inf": 12.37,
    # Derived from the paper's wp=2.7396e14 rad/s with m*=0.033 m_e.
    "carrier_density_cm3": 7.782250266240598e17,
    "effective_mass_ratio": 0.033,
    "damping_rate_rad_s": 1.55e11,
}
al_parameters = {
    "eps_inf": 1.0,
    "plasma_frequency_rad_s": 2.24e16,
    "damping_rate_rad_s": 1.24e14,
}

theta_deg = 7.0
phi_deg = 0.0
polarizations = ("TE", "TM")
angles_deg = (+theta_deg, -theta_deg)
# Circular (11, 11) retains 377 harmonics. The TE peak changes by only 0.2%
# from (9, 9) to (11, 11); the tiny circular hole makes TM converge more slowly.
orders = (5, 5)
workers = 14
# Fourier convergence controls. "Default" preserves the original example result.
workers_per_angle = max(1, workers // len(angles_deg))

wavelength_min_um = 15.3
wavelength_max_um = 15.9
wavelength_point_count = 301
wavelengths_um = np.linspace(wavelength_min_um, wavelength_max_um, wavelength_point_count)

paper_peak_um = {"TE": 15.630, "TM": 15.658}
paper_peak_eta = {"TE": 0.9232, "TM": 0.9146}
paper_markers = {pol: (paper_peak_um[pol],) for pol in polarizations}

plot_path = project_root / "images" / "liu_2025_ge_circular_hole_inas_fig2.png"
figure_size = (12.0, 4.8)
figure_dpi = 240
show_plot = False

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
    sim.SetMaterial("SiO2", sio2_refractive_index**2 + 0.0j)
    sim.SetMaterialModel("InAs", "InAsMagneto", inas_parameters)
    sim.SetMaterialModel("Al", "Drude", al_parameters)
    sim.SetSuperstrate("Air")
    sim.SetSubstrate("SiO2")

    # Exact rectangle/circle coefficients feed the ordered Li
    # blocks; no sampled material-permittivity transform is used.
    sim.AddLayer("Ge_square_with_circular_hole", ge_height_um, "Air")
    sim.SetRegionRectangle(
        "Ge_square_with_circular_hole",
        "Ge",
        Center=(0.0, 0.0),
        Angle=0.0,
        Halfwidths=(0.5 * ge_square_width_um, 0.5 * ge_square_width_um),
    )
    sim.SetRegionCircle(
        "Ge_square_with_circular_hole",
        "Air",
        Center=(0.0, 0.0),
        Radius=0.5 * circular_hole_diameter_um,
    )
    sim.AddLayer("InAs", inas_height_um, "InAs")
    sim.AddLayer("Al_reflector", al_height_um, "Al")
    return sim


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
    fig.suptitle(f"Ge square/circular-hole array / InAs / Al, orders={orders}")
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(plot_path, dpi=figure_dpi)
    print(f"Saved {plot_path}.")
    plt.show() if show_plot else plt.close(fig)


def main() -> np.ndarray:
    print(f"Using asyrcwa backend from {asyrcwa.module_directory()}.")
    with asyrcwa.timed_step("solve Liu et al. Fig. 2 TE/TM spectrum"):
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
        for pol in polarizations:
            rows = spectrum[spectrum["pol"] == pol]
            row = rows[int(np.argmin(abs(rows["lambda_um"] - paper_peak_um[pol])))]
            print(
                f"{pol}: eta({paper_peak_um[pol]:.3f} um)={row['eta']:.4f}; "
                f"paper {paper_peak_eta[pol]:.4f}"
            )
    with asyrcwa.timed_step("plot spectrum"):
        plot_spectrum(spectrum)
    return spectrum


if __name__ == "__main__":
    main()
