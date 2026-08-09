"""Reproduce Zou et al., IJHMT 231 (2024) 125819, Fig. 2.

The paper studies a two-dimensional Si-cylinder grating on an InAs
magneto-optical film and an Ag reflector.  The spectra are calculated with
the local RCWA backend using the paper's definitions
alpha(theta) = 1 - R(theta) - T(theta),
e(theta) = 1 - R(-theta) - T(-theta), and
eta = |alpha - e|.
"""

from __future__ import annotations

from pathlib import Path

project_root = Path(__file__).resolve().parents[1]

import matplotlib
import numpy as np

import asyrcwa
from asyrcwa.spectrum import plot_fr_axes, solve_fr, summarize_fr


# Zou et al., Fig. 1 and Table 1.
period_um = 5.33
silicon_cylinder_diameter_um = 2.13
silicon_cylinder_height_um = 5.94
inas_height_um = 1.962
ag_height_um = 0.50
silicon_refractive_index = 3.48

# The paper gives the InAs plasma frequency in Eq. (4).  The repository's
# built-in InAsMagneto model takes carrier density and effective mass instead;
# this density is the equivalent value for wp=2.7396e14 rad/s and m*=0.033 me.
inas_plasma_frequency_rad_s = 2.7396e14
inas_parameters = {
    "magnetic_field_t": +3.0,
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

theta_deg = 33.0
phi_deg = 0.0
polarizations = ("TE", "TM")
angles_deg = (+theta_deg, -theta_deg)

# Circular truncation is used for the genuinely 2D cylindrical grating.
# This is a retained-order setting; the paper's convergence order was not
# specified in the article, so the value is kept explicit for reproducibility.
orders = (5, 5)
workers = 20
# Fourier convergence controls. "Default" preserves the original example result.
workers_per_angle = max(1, workers // len(angles_deg))

wavelength_min_um = 11.55
wavelength_max_um = 12.05
wavelength_point_count = 401
wavelengths_um = np.linspace(
    wavelength_min_um,
    wavelength_max_um,
    wavelength_point_count,
)

paper_reference_wavelength_um = 11.763
paper_reference_values = {
    "TE": {"A": 0.037, "e": 0.973, "eta": 0.936},
    "TM": {"A": 0.983, "e": 0.043, "eta": 0.940},
}
paper_peak_wavelengths_um = {
    "TE": (11.763, 11.840),
    "TM": (11.763, 11.940),
}
paper_markers = {
    polarization: paper_peak_wavelengths_um[polarization]
    for polarization in polarizations
}

plot_path = (
    project_root
    / "images"
    / "zou_2024_polarization_independent_cylindrical_grating_fig2.png"
)
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
    sim.SetMaterialModel("Si", "Si")
    sim.SetMaterialModel("InAs", "InAsMagneto", inas_parameters)
    sim.SetMaterialModel("Ag", "Drude", ag_parameters)
    sim.SetSuperstrate("Air")
    # The 0.5-um Ag reflector is treated as the opaque bottom layer; Air is
    # used below it because the paper does not specify a separate substrate.
    sim.SetSubstrate("Air")

    # The circle uses RETICOLO's area-normalized nested-rectangle geometry;
    # direct and ordered-Li blocks therefore share exactly the same cells.
    sim.AddLayer("Si_cylinders", silicon_cylinder_height_um, "Air")
    sim.SetRegionCircle(
        "Si_cylinders",
        "Si",
        Center=(0.0, 0.0),
        Radius=0.5 * silicon_cylinder_diameter_um,
    )
    sim.AddLayer("InAs", inas_height_um, "InAs")
    sim.AddLayer("Ag", ag_height_um, "Ag")
    return sim


def nearest_row(rows: np.ndarray, wavelength_um: float) -> np.void:
    return rows[int(np.argmin(abs(rows["lambda_um"] - wavelength_um)))]


def print_paper_comparison(spectrum: np.ndarray) -> None:
    for polarization in polarizations:
        rows = spectrum[spectrum["pol"] == polarization]
        reference = nearest_row(rows, paper_reference_wavelength_um)
        expected = paper_reference_values[polarization]
        print(
            f"{polarization} at {paper_reference_wavelength_um:.3f} um: "
            f"A/e/eta={reference['A']:.4f}/"
            f"{reference['emissivity_same_channel']:.4f}/"
            f"{reference['eta']:.4f}; paper="
            f"{expected['A']:.3f}/{expected['e']:.3f}/{expected['eta']:.3f}"
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
        "Si cylindrical grating / InAs / Ag, "
        f"B=3 T, theta=+/-{theta_deg:g} deg, orders={orders}"
    )
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(plot_path, dpi=figure_dpi)
    print(f"Saved {plot_path}.")
    plt.show() if show_plot else plt.close(fig)


def main() -> np.ndarray:
    print(f"Using asyrcwa backend from {asyrcwa.module_directory()}.")
    print(
        f"period={period_um:g} um, diameter={silicon_cylinder_diameter_um:g} um, "
        f"B={inas_parameters['magnetic_field_t']:g} T, "
        f"wp(InAs)={inas_plasma_frequency_rad_s:.6e} rad/s"
    )
    with asyrcwa.timed_step("solve Zou et al. Fig. 2 spectrum"):
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
