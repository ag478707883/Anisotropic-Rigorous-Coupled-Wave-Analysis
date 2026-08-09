"""Reproduce the embedded-InAs-grating spectrum in Zou et al. (2025), Fig. 3.

Article: "Wide-angle strong nonreciprocal thermal radiation with the embedded
InAs grating".
Journal: Optical Materials 167 (2025) 117277.
DOI: 10.1016/j.optmat.2025.117277.

The paper uses FEM. This example evaluates the same one-dimensional periodic
stack with the repository RCWA backend and the paper definitions
alpha(theta) = 1 - R(theta) and e(theta) = 1 - R(-theta).
"""

from __future__ import annotations

from pathlib import Path

project_root = Path(__file__).resolve().parents[1]

import matplotlib
import numpy as np

import asyrcwa
from asyrcwa.spectrum import (
    plot_fr_axes,
    solve_fr,
    summarize_fr,
)


# Paper geometry: Fig. 1. The structure is periodic along paper x, invariant
# along paper z, and stacked along paper y. Solver x is periodic, solver y is
# invariant, and solver +z follows AddLayer order from Air toward the substrate.
period_um = 7.24
strip_width_um = 2.00
grating_width_um = 3.20

strip_height_um = 0.40
buffer_separation_height_um = 0.40
grating_ridge_height_um = 0.60
inas_base_height_um = 0.40
strip_embedding_depth_um = 0.02
al_reflector_height_um = 0.20
strip_exposed_height_um = strip_height_um - strip_embedding_depth_um

# X-Z solver section through one period; finite layers follow top-to-bottom:
#
#        Air superstrate
#        +--------- InAs strip ---------+  exposed 0.38 um, width w1
#        | Air                          |
#        +---- embedded InAs/SiO2 ------+  0.02 um
#        |             SiO2             |  0.40 um
#        +------ InAs/SiO2 grating -----+  0.60 um, width w2
#        |             InAs             |  0.40 um
#        |              Al              |  0.20 um
#        +------------------------------+
#                  SiO2 substrate

sio2_refractive_index = 1.45
inas_parameters = {
    "magnetic_field_t": 1.0,
    "eps_inf": 12.37,
    "carrier_density_cm3": 7.8e17,
    "effective_mass_ratio": 0.033,
    "damping_rate_rad_s": 1.55e11,
}
al_drude_parameters = {
    "eps_inf": 1.0,
    "plasma_frequency_rad_s": 2.24e16,
    "damping_rate_rad_s": 1.24e14,
}

theta_deg = 60.0
phi_deg = 0.0
polarizations = ("TM",)
angles_deg = (+theta_deg, -theta_deg)

# Every patterned slice is a 1D lamellar grating. The backend therefore uses
# analytic Fourier coefficients with directional Li inverse factorization.
orders = (17, 0)
workers = 10
# Fourier convergence controls. "Default" preserves the original example result.
workers_per_angle = max(1, workers // len(angles_deg))

wavelength_min_um = 25.20
wavelength_max_um = 26.00
wavelength_point_count = 801
wavelengths_um = np.linspace(
    wavelength_min_um,
    wavelength_max_um,
    wavelength_point_count,
)

paper_emission_peak_um = 25.40
paper_emission_peak_value = 0.995
paper_absorptivity_at_emission_peak = 0.082
paper_eta_at_emission_peak = 0.913
paper_absorption_peak_um = 25.82
paper_absorption_peak_value = 0.997
paper_emissivity_at_absorption_peak = 0.060
paper_eta_at_absorption_peak = 0.937
paper_reference_markers = {
    "TM": (paper_emission_peak_um, paper_absorption_peak_um),
}

plot_path = project_root / "images" / "zou_2025_embedded_inas_grating_fig3.png"
figure_size = (7.2, 4.8)
figure_dpi = 240
show_plot = True

if not show_plot:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt


def build_simulation(rcwa: object) -> object:
    simulation = rcwa.New(
        Lattice=period_um,
        Orders=orders,
    )
    simulation.SetMaterial("Air", 1.0 + 0.0j)
    simulation.SetMaterial("SiO2", sio2_refractive_index**2 + 0.0j)
    simulation.SetMaterialModel("InAs", "InAsMagneto", inas_parameters)
    simulation.SetMaterialModel("Al", "Drude", al_drude_parameters)
    simulation.SetSuperstrate("Air")
    simulation.SetSubstrate("SiO2")

    simulation.AddLayer("InAs_strip_exposed", strip_exposed_height_um, "Air")
    simulation.SetRegionRectangle(
        "InAs_strip_exposed",
        "InAs",
        Center=(0.0, 0.0),
        Angle=0.0,
        Halfwidths=(0.5 * strip_width_um, 0.5 * period_um),
    )
    simulation.AddLayer("InAs_strip_embedded", strip_embedding_depth_um, "SiO2")
    simulation.SetRegionRectangle(
        "InAs_strip_embedded",
        "InAs",
        Center=(0.0, 0.0),
        Angle=0.0,
        Halfwidths=(0.5 * strip_width_um, 0.5 * period_um),
    )
    simulation.AddLayer(
        "SiO2_buffer",
        buffer_separation_height_um,
        "SiO2",
    )
    simulation.AddLayer("InAs_grating", grating_ridge_height_um, "SiO2")
    simulation.SetRegionRectangle(
        "InAs_grating",
        "InAs",
        Center=(0.0, 0.0),
        Angle=0.0,
        Halfwidths=(0.5 * grating_width_um, 0.5 * period_um),
    )
    simulation.AddLayer("InAs_base", inas_base_height_um, "InAs")
    simulation.AddLayer("Al_reflector", al_reflector_height_um, "Al")
    return simulation


def nearest_row(rows: np.ndarray, wavelength_um: float) -> np.void:
    return rows[int(np.argmin(np.abs(rows["lambda_um"] - wavelength_um)))]


def print_paper_comparison(spectrum: np.ndarray) -> None:
    rows = spectrum[spectrum["pol"] == "TM"]
    absorption_peak = rows[int(np.argmax(rows["A"]))]
    emission_peak = rows[int(np.argmax(rows["emissivity_same_channel"]))]
    eta_peak = rows[int(np.argmax(rows["eta"]))]
    at_paper_emission = nearest_row(rows, paper_emission_peak_um)
    at_paper_absorption = nearest_row(rows, paper_absorption_peak_um)

    print(
        "  computed absorption peak: "
        f"{float(absorption_peak['lambda_um']):.4f} um, "
        f"alpha={float(absorption_peak['A']):.4f}; "
        f"paper {paper_absorption_peak_um:.2f} um, "
        f"alpha={paper_absorption_peak_value:.3f}"
    )
    print(
        "  computed emission peak: "
        f"{float(emission_peak['lambda_um']):.4f} um, "
        f"e={float(emission_peak['emissivity_same_channel']):.4f}; "
        f"paper {paper_emission_peak_um:.2f} um, "
        f"e={paper_emission_peak_value:.3f}"
    )
    print(
        f"  computed max eta={float(eta_peak['eta']):.4f} at "
        f"{float(eta_peak['lambda_um']):.4f} um"
    )
    print(
        f"  at paper {paper_emission_peak_um:.2f} um: "
        f"alpha/e/eta={float(at_paper_emission['A']):.4f}/"
        f"{float(at_paper_emission['emissivity_same_channel']):.4f}/"
        f"{float(at_paper_emission['eta']):.4f} "
        f"(paper {paper_absorptivity_at_emission_peak:.3f}/"
        f"{paper_emission_peak_value:.3f}/{paper_eta_at_emission_peak:.3f})"
    )
    print(
        f"  at paper {paper_absorption_peak_um:.2f} um: "
        f"alpha/e/eta={float(at_paper_absorption['A']):.4f}/"
        f"{float(at_paper_absorption['emissivity_same_channel']):.4f}/"
        f"{float(at_paper_absorption['eta']):.4f} "
        f"(paper {paper_absorption_peak_value:.3f}/"
        f"{paper_emissivity_at_absorption_peak:.3f}/"
        f"{paper_eta_at_absorption_peak:.3f})"
    )
    print(
        f"  max forward/reverse transmittance="
        f"{max(float(np.max(rows['T'])), float(np.max(rows['T_reverse']))):.3e}"
    )


def plot_spectrum(spectrum: np.ndarray) -> None:
    figure, axis = plt.subplots(
        1,
        1,
        figsize=figure_size,
        constrained_layout=True,
    )
    axes = np.asarray([axis])
    plot_fr_axes(
        axes,
        spectrum,
        polarizations,
        markers=paper_reference_markers,
    )
    axis.set_xlim(wavelength_min_um, wavelength_max_um)
    figure.suptitle(
        f"Embedded InAs grating, B=1 T, theta={theta_deg:g} deg, orders={orders}"
    )
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(plot_path, dpi=figure_dpi)
    print(f"Saved {plot_path}.")
    plt.show() if show_plot else plt.close(figure)


def main() -> np.ndarray:
    print(f"Using asyrcwa backend from {asyrcwa.module_directory()}.")
    print(
        f"Orders={orders}, workers={workers}, "
        f"theta=+/-{theta_deg:g} deg, B_solver="
        f"{inas_parameters['magnetic_field_t']:g} T."
    )
    with asyrcwa.timed_step("solve forward/reverse spectrum"):
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
